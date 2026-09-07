"""
Ingestion pipeline: reads transcript files from TRANSCRIPTS_DIR and
stores them as searchable chunks in a local ChromaDB vector database.

Supports:
  - *.json  — Whisper output (has 'segments' array with timestamps)
  - *.txt   — plain text transcripts (chunked by word count)

Run once, and re-run whenever new transcript files are added.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# Falls back to the transcripts/ folder bundled in this repo (used on Streamlit
# Cloud, where there's no .env). Locally, .env overrides this with the full
# Desktop Archive path.
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", str(Path(__file__).parent / "transcripts")))
# Supplementary documents (PDFs converted to .txt) that aren't class recordings —
# e.g. campaign plans, content calendars. Tagged source_type="reference" so the
# app never suggests them as a "class to watch".
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", str(Path(__file__).parent / "reference")))
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "class_transcripts"
CHUNK_SIZE = 15          # segments per chunk (JSON mode)
CHUNK_OVERLAP = 3        # segment overlap (JSON mode)
TXT_CHUNK_WORDS = 300    # words per chunk (TXT mode)
TXT_CHUNK_OVERLAP = 50   # word overlap (TXT mode)
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json_transcript(path: Path) -> list:
    """Returns list of chunks from a Whisper JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        return []
    return _chunk_segments(segments, CHUNK_SIZE, CHUNK_OVERLAP)


def load_txt_transcript(path: Path) -> list:
    """Returns list of chunks from a plain text transcript file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return _chunk_words(words, TXT_CHUNK_WORDS, TXT_CHUNK_OVERLAP)


# ── Chunking helpers ──────────────────────────────────────────────────────────

def _chunk_segments(segments: list, chunk_size: int, overlap: int) -> list:
    """Sliding window over Whisper segment objects."""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(segments), step):
        window = segments[i : i + chunk_size]
        if not window:
            break
        text = " ".join(seg["text"].strip() for seg in window)
        chunks.append(
            {
                "text": text,
                "start_time": window[0]["start"],
                "end_time": window[-1]["end"],
                "timestamp_label": (
                    f"{_fmt_ts(window[0]['start'])} – {_fmt_ts(window[-1]['end'])}"
                ),
            }
        )
    return chunks


def _chunk_words(words: list, chunk_size: int, overlap: int) -> list:
    """Sliding window over a flat word list (for plain text files)."""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if not window:
            break
        chunks.append(
            {
                "text": " ".join(window),
                "start_time": 0.0,
                "end_time": 0.0,
                "timestamp_label": f"part {len(chunks) + 1}",
            }
        )
    return chunks


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_from_dir(transcripts_dir: Path, model, collection, source_type="class", log=print) -> int:
    """Chunks + embeds every transcript in transcripts_dir into collection.
    Idempotent — re-running replaces any existing chunks for a given source file.
    source_type is tagged on every chunk's metadata ("class" for session
    recordings, "reference" for supplementary documents).
    Returns the number of chunks added this run."""
    json_files = sorted(transcripts_dir.glob("*.json"))
    txt_files = sorted(transcripts_dir.glob("*.txt"))
    all_files = json_files + txt_files

    if not all_files:
        log(f"No transcript files found in {transcripts_dir}")
        log("Add *.json (Whisper) or *.txt files and re-run.")
        return 0

    log(f"Found {len(json_files)} JSON + {len(txt_files)} TXT files\n")
    total_chunks = 0

    for filepath in all_files:
        source_name = filepath.stem
        log(f"Processing: {filepath.name}")

        # Remove existing docs for this file (idempotent re-runs)
        existing = collection.get(where={"source_file": source_name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            log(f"  Removed {len(existing['ids'])} existing chunks")

        if filepath.suffix == ".json":
            chunks = load_json_transcript(filepath)
        else:
            chunks = load_txt_transcript(filepath)

        if not chunks:
            log(f"  No content found, skipping.")
            continue

        log(f"  → {len(chunks)} chunks")
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_file": source_name,
                "source_type": source_type,
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "timestamp_label": c["timestamp_label"],
            }
            for c in chunks
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        total_chunks += len(chunks)
        log(f"  Stored {len(chunks)} chunks for '{source_name}'\n")

    log(
        f"Done. Total chunks in collection: {collection.count()} "
        f"(added {total_chunks} this run)"
    )
    return total_chunks


def ingest():
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ingest_from_dir(TRANSCRIPTS_DIR, model, collection, source_type="class")
    ingest_from_dir(REFERENCE_DIR, model, collection, source_type="reference")


if __name__ == "__main__":
    ingest()
