"""
Streamlit chat UI for the LvR Mentoring LvR GPT.
Password-gated, RAG-powered, responds in the student's language.
"""

import base64
import hashlib
import hmac
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import extra_streamlit_components as stx
import httpx
import streamlit as st
from dotenv import load_dotenv
import chromadb
import anthropic

from embeddings import embed_texts
from ingest import ingest_from_dir

load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
TRANSCRIPTS_DIR = Path(os.environ.get("TRANSCRIPTS_DIR", str(Path(__file__).parent / "transcripts")))
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR", str(Path(__file__).parent / "reference")))
STICKER_PATH = Path(__file__).parent / "assets" / "lvr-sticker.png"
COLLECTION_NAME = "class_transcripts"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 5

# Vimeo OTT (VHX) — the product bundle that includes LvR GPT access
VHX_PRODUCT_ID = os.environ.get("VHX_PRODUCT_ID", "264536")
VHX_PURCHASE_URL = "https://lvrmentoringondemand.vhx.tv"

# Private content repo — the code repo is public, so class transcripts and
# reference docs live in a separate private repo, cloned at boot if missing.
CONTENT_REPO = os.environ.get("CONTENT_REPO", "lumonfortlvr/lvr-gpt-content")

# Magic-link login
APP_URL = os.environ.get("APP_URL", "http://localhost:8502")
MAGIC_LINK_TTL = 15 * 60  # 15 minutes
REMEMBER_ME_TTL = 30 * 24 * 60 * 60  # 30 days
REMEMBER_COOKIE_NAME = "lvrgpt_session"


def _get_secret(key: str) -> str:
    # Try os.environ first (set by load_dotenv above)
    val = os.environ.get(key, "")
    if val:
        return val
    # Fallback: read .env file directly
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line[len(key) + 1:].strip()
    # Fallback: Streamlit Cloud secrets
    try:
        val = st.secrets.get(key, "")
    except Exception:
        pass
    return val or ""


SYSTEM_PROMPT = """\
You are a helpful teaching assistant for Leticia Van Riel's mentoring program.
Your job is to answer students' questions based ONLY on the excerpts provided \
below — class transcripts, plus supplementary program documents such as \
release campaign plans and content calendars.

Content & Knowledge:
- Answer only from the excerpts provided — never invent information.
- If the answer is not found in the excerpts, say so honestly.
- Do not answer questions unrelated to the class material.
- Never cite specific sources (e.g. never say "In Session 5, Leticia said...") \
— answer as general program knowledge, not as a citation.

Privacy:
- Never mention Jessica by name.
- Never name the special guest — refer to them only as "the special guest."
- Never quote or name a specific student — protect student privacy from group \
Q&A sessions; present anything a student said as general guidance instead.

Voice & Language:
- Always respond in English, regardless of what language the student writes in.
- Write in Leticia's voice — direct, warm, and grounded in real experience, \
never corporate or academic.
  - Personality: no-nonsense and blunt when something isn't working, but \
always paired with genuine care — never cold or preachy. Motivate through \
realism ("this industry is a mess, but the reward is worth it"), not empty \
positivity.
  - Speech patterns: use her characteristic check-in phrases naturally, but \
not in every sentence — okay?, alright?, you know?, right?, guys, I mean.
  - Sentence style: short-to-medium sentences; think out loud, sometimes \
rephrasing mid-thought before landing the point.
  - Avoid: corporate language, vague platitudes, academic/theoretical \
framing, moralising repetition, and dashes (em dashes or hyphens used as \
punctuation) — write in plain sentences or split into two instead.
  - Draw on recurring themes when relevant: treat yourself as a business, \
hustle/chasing is non-negotiable, networking matters, online presence is a \
first impression, persistence over annoyance.
  - Strong language (e.g. "fuck," "shit") is allowed only very sparingly, \
for emphasis — not a default habit.
- Be concise and direct. You may quote or paraphrase the source content. Skip \
filler openers like "Great question!", "I'd be happy to help", or "Certainly!" \
— just answer.\
"""

# ── Global CSS (LvR brand) ────────────────────────────────────────────────────
# One quiet, graphite interface (ChatGPT-like restraint) with a single loud
# moment — the LvR sticker — carrying all the brand personality. Everything
# else stays sentence case; no text-transform is applied to brand text, so
# "LvR" always renders with its lowercase v.
LVR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&display=swap');

:root {
    --lvr-bg: #1c1c1e;
    --lvr-bg-sidebar: rgba(22,22,24,0.78);
    --lvr-surface: #2a2a2d;
    --lvr-surface-glass: rgba(255,255,255,0.055);
    --lvr-surface-hover: #323235;
    --lvr-border: rgba(255,255,255,0.08);
    --lvr-border-strong: rgba(255,255,255,0.18);
    --lvr-border-glass: rgba(255,255,255,0.14);
    --lvr-text: #ececec;
    --lvr-text-muted: #8e8e93;
    --lvr-text-faint: #6b6b6f;
    --lvr-accent: #FDE90C;
    --lvr-font: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Archivo', sans-serif;
}

/* ── Base — a soft keynote-stage spotlight, not a flat fill ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(ellipse 1100px 620px at 50% -8%, #3d3d43 0%, #232326 42%, #18181a 78%, #141416 100%) var(--lvr-bg) !important;
    background-attachment: fixed !important;
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
    letter-spacing: -0.006em;
}
[data-testid="stMain"], .main, section.main {
    background: transparent !important;
}
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* ── Sidebar — frosted glass over the gradient ── */
[data-testid="stSidebar"] {
    background-color: var(--lvr-bg-sidebar) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--lvr-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--lvr-border) !important;
}

/* Sidebar sticker + wordmark */
.st-key-sticker-sidebar { display: flex; justify-content: center; }
.st-key-sticker-sidebar [data-testid="stImage"] img {
    width: 42px !important;
    height: 42px !important;
    transform: rotate(-8deg);
    filter: drop-shadow(0 3px 6px rgba(0,0,0,0.5));
}
.sidebar-wordmark {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--lvr-text) !important;
    text-align: center;
    margin-top: 0.5rem;
}
.sidebar-tag {
    font-size: 0.7rem;
    color: var(--lvr-text-faint) !important;
    text-align: center;
    margin-top: 0.1rem;
}

/* New-chat button: quiet, bordered — not a shouting CTA */
.st-key-new-chat .stButton > button {
    background-color: transparent !important;
    color: var(--lvr-text) !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    border: 1px solid var(--lvr-border-strong) !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    box-shadow: none !important;
}
.st-key-new-chat .stButton > button:hover {
    background-color: var(--lvr-surface) !important;
}

/* Chat history list (ghost rows) */
.st-key-chat-list .stButton > button {
    background-color: transparent !important;
    color: var(--lvr-text-muted) !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 0.6rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.st-key-chat-list .stButton > button:hover {
    background-color: var(--lvr-surface) !important;
    color: var(--lvr-text) !important;
}
.st-key-chat-list .stButton > button:focus:not(:active) {
    border-color: transparent !important;
}

/* ── Headings — sentence case, no forced uppercase ── */
h1, h2, h3 {
    font-family: var(--lvr-font) !important;
    color: var(--lvr-text) !important;
    font-weight: 700;
    letter-spacing: normal;
    text-transform: none;
}

/* ── Chat messages — ChatGPT-like: assistant is plain text, user gets a quiet bubble ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.35rem 0 !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background-color: var(--lvr-surface) !important;
    border-radius: 16px !important;
    padding: 0.7rem 1.1rem !important;
    display: inline-block;
    max-width: 80%;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
    line-height: 1.6;
}

/* ── Chat input — frosted glass, floating over the gradient ── */
[data-testid="stChatInput"] {
    background-color: var(--lvr-surface-glass) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--lvr-border-glass) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255,255,255,0.28) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.32) !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--lvr-text-faint) !important;
}

/* ── Buttons — quiet graphite with a glossy top highlight (Aqua-era polish) ── */
.stButton > button {
    background: linear-gradient(180deg, var(--lvr-surface-hover) 0%, var(--lvr-surface) 100%) !important;
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    border: 1px solid var(--lvr-border) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 1px 2px rgba(0,0,0,0.3) !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(180deg, #3a3a3e 0%, var(--lvr-surface-hover) 100%) !important;
    border-color: var(--lvr-border-strong) !important;
}

/* Unlock button: the one deliberate accent moment on the login screen */
.st-key-unlock-btn .stButton > button {
    background: linear-gradient(180deg, #ffef6b 0%, var(--lvr-accent) 55%, #e6d40b 100%) !important;
    color: #16160a !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.55), 0 6px 18px rgba(253,233,12,0.3) !important;
}
.st-key-unlock-btn .stButton > button:hover {
    background: linear-gradient(180deg, #fff48a 0%, #ffee4d 55%, #ecdc12 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 8px 22px rgba(253,233,12,0.4) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background-color: var(--lvr-surface) !important;
    color: var(--lvr-text) !important;
    border: 1px solid var(--lvr-border) !important;
    border-radius: 8px !important;
    font-family: var(--lvr-font) !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--lvr-border-strong) !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--lvr-text-faint) !important;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background-color: var(--lvr-surface) !important;
    border: 1px solid var(--lvr-border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--lvr-text) !important;
    font-family: var(--lvr-font) !important;
    font-size: 0.85rem !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: var(--lvr-text-muted) !important;
    font-size: 0.85rem !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--lvr-border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--lvr-bg); }
::-webkit-scrollbar-thumb { background: #46464a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5a5a5e; }

/* ── Caption / small text ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: var(--lvr-text-faint) !important;
    font-family: var(--lvr-font) !important;
}

/* ── Error / warning ── */
[data-testid="stAlert"] {
    background-color: #2a1c1c !important;
    border-left: 3px solid #ff6b6b !important;
    color: #ffb3b3 !important;
    border-radius: 8px;
}

/* ── Login page specific ── */

/* The glass panel the whole login form floats in — Apple-style translucent
   material catching the spotlight gradient behind it */
.st-key-login-card {
    background: var(--lvr-surface-glass) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border: 1px solid var(--lvr-border-glass) !important;
    border-top: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 22px !important;
    padding: 2.5rem 2.25rem 2rem !important;
    margin-top: 3rem !important;
    box-shadow: 0 30px 70px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.lvr-login-wrapper {
    text-align: center;
}
.st-key-sticker-login {
    display: flex;
    justify-content: center;
    margin-bottom: 1.25rem;
    position: relative;
}
.st-key-sticker-login::before {
    content: "";
    position: absolute;
    top: -30px;
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, rgba(253,233,12,0.22) 0%, rgba(253,233,12,0) 70%);
    pointer-events: none;
}
.st-key-sticker-login [data-testid="stImage"] img {
    width: 96px !important;
    height: 96px !important;
    transform: rotate(-7deg);
    filter: drop-shadow(0 10px 18px rgba(0,0,0,0.55));
    position: relative;
}
.lvr-login-logo {
    font-family: var(--lvr-font);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--lvr-text);
    line-height: 1;
    margin-bottom: 0.4rem;
}
.lvr-login-sub {
    font-family: var(--lvr-font);
    font-size: 0.85rem;
    color: var(--lvr-text-muted);
    margin-bottom: 2.25rem;
}
.lvr-login-label {
    font-family: var(--lvr-font);
    font-size: 0.8rem;
    color: var(--lvr-text-muted);
    margin-bottom: 0.5rem;
    text-align: left;
}
.lvr-buy-link {
    display: block;
    margin-top: 0.75rem;
    font-family: var(--lvr-font);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--lvr-accent);
    text-decoration: none;
}
.lvr-buy-link:hover {
    text-decoration: underline;
}
</style>
"""


# ── Access gate — Vimeo OTT (VHX) subscription check ──────────────────────────
def _is_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()))


def verify_subscriber(email: str) -> bool:
    """Checks the Vimeo OTT (VHX) API for an active LvR Mentoring ON Demand
    subscription under this email. Fails closed on any API/network error."""
    api_key = _get_secret("VHX_API_KEY")
    if not api_key:
        return False
    try:
        resp = httpx.get(
            "https://api.vhx.tv/customers",
            params={"query": email.strip()},
            auth=(api_key, ""),
            timeout=10,
        )
        resp.raise_for_status()
        customers = resp.json().get("_embedded", {}).get("customers", [])
    except httpx.HTTPError:
        return False

    target = email.strip().lower()
    for customer in customers:
        if customer.get("email", "").strip().lower() != target:
            continue
        products = customer.get("_embedded", {}).get("products", [])
        if any(str(p.get("id")) == str(VHX_PRODUCT_ID) for p in products):
            return True
    return False


def _sign(payload: str) -> str:
    secret = _get_secret("SECRET_KEY")
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def make_token(email: str, ttl_seconds: int) -> str:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{email.strip().lower()}:{expires_at}"
    raw = f"{payload}:{_sign(payload)}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8")


def verify_token(token: str):
    """Returns the email if the token is validly signed and unexpired, else None."""
    try:
        raw = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        email, expires_at, signature = raw.rsplit(":", 2)
    except Exception:
        return None
    payload = f"{email}:{expires_at}"
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    if int(expires_at) < int(time.time()):
        return None
    return email


def get_cookie_manager():
    # Not cached: CookieManager renders a component (widget-like) on every
    # run to report cookie values back — caching it triggers Streamlit's
    # CachedWidgetWarning and only reads cookies on a cache miss.
    return stx.CookieManager(key="cookie_manager")


def send_magic_link_email(email: str, link: str) -> bool:
    api_key = _get_secret("RESEND_API_KEY")
    if not api_key:
        return False
    from_addr = _get_secret("RESEND_FROM_EMAIL") or "LvR Mentoring <onboarding@resend.dev>"
    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": from_addr,
                "to": [email],
                "subject": "Your LvR GPT login link",
                "html": (
                    "<p>Tap below to open LvR GPT — this link works for 15 minutes.</p>"
                    f'<p><a href="{link}">Open LvR GPT →</a></p>'
                ),
            },
            timeout=10,
        )
        return resp.status_code < 300
    except httpx.HTTPError:
        return False


def request_login(value: str):
    """Handles a login-screen submission. Returns a dict with at least a
    "status" key: "admin" (grant immediately), "sent" (magic link emailed),
    "send_failed" (couldn't email — includes a "dev_link" to test with while
    RESEND_API_KEY isn't configured yet), "invalid_email", or "not_subscriber"."""
    value = value.strip()
    admin_password = _get_secret("ADMIN_PASSWORD")
    if admin_password and hmac.compare_digest(value.encode("utf-8"), admin_password.encode("utf-8")):
        return {"status": "admin"}
    if not _is_email(value):
        return {"status": "invalid_email", "message": "Enter the email you used to buy LvR Mentoring ON Demand."}
    if not verify_subscriber(value):
        return {
            "status": "not_subscriber",
            "message": "We couldn't find an active LvR Mentoring ON Demand subscription for that email.",
        }
    token = make_token(value, MAGIC_LINK_TTL)
    link = f"{APP_URL}/?token={token}"
    if send_magic_link_email(value, link):
        return {"status": "sent", "message": f"Check your inbox — we sent a login link to {value}."}
    return {
        "status": "send_failed",
        "message": "Couldn't send the login email — RESEND_API_KEY isn't configured yet.",
        "dev_link": link,
    }


def try_auto_login(cookie_manager):
    """Call once near the top of main(): logs the user in from a magic-link
    token in the URL, or from a remembered device cookie, without showing
    the login screen."""
    if st.session_state.get("authenticated"):
        return

    token = st.query_params.get("token")
    if token:
        email = verify_token(token)
        st.query_params.clear()
        if email and verify_subscriber(email):
            st.session_state.authenticated = True
            cookie_manager.set(
                REMEMBER_COOKIE_NAME,
                make_token(email, REMEMBER_ME_TTL),
                max_age=REMEMBER_ME_TTL,
            )
            st.rerun()
        return

    remembered = cookie_manager.get(REMEMBER_COOKIE_NAME)
    if remembered:
        email = verify_token(remembered)
        if email and verify_subscriber(email):
            st.session_state.authenticated = True


def show_login():
    st.markdown(LVR_CSS, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(key="login-card"):
            st.markdown('<div class="lvr-login-wrapper">', unsafe_allow_html=True)
            with st.container(key="sticker-login"):
                st.image(str(STICKER_PATH))
            st.markdown(
                """
                <div class="lvr-login-logo">LvR GPT</div>
                <div class="lvr-login-sub">The straight-to-the-point smart chat trained on the LvR lessons archive.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<p class="lvr-login-label">Email address</p>', unsafe_allow_html=True)
            email = st.text_input(
                "email",
                key="login_email",
                label_visibility="collapsed",
                placeholder="you@example.com",
            )
            with st.container(key="unlock-btn"):
                if st.button("Send login link", use_container_width=True):
                    result = request_login(email)
                    status = result["status"]
                    if status == "admin":
                        st.session_state.authenticated = True
                        st.rerun()
                    elif status == "sent":
                        st.success(result["message"])
                    elif status == "send_failed":
                        st.warning(result["message"])
                        st.markdown(
                            f'<a href="{result["dev_link"]}" target="_blank" '
                            f'class="lvr-buy-link">Open login link (dev mode) →</a>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(result["message"])
                        if status == "not_subscriber":
                            st.markdown(
                                f'<a href="{VHX_PURCHASE_URL}" target="_blank" '
                                f'class="lvr-buy-link">Get LvR Mentoring ON Demand →</a>',
                                unsafe_allow_html=True,
                            )


def ensure_content_downloaded():
    """Clones the private transcripts/reference content repo if it's not
    already present locally — needed on Streamlit Cloud, where only the
    public code repo is checked out and the course content lives separately."""
    if TRANSCRIPTS_DIR.exists() and REFERENCE_DIR.exists():
        return
    token = _get_secret("CONTENT_REPO_TOKEN")
    if not token:
        raise RuntimeError(
            "CONTENT_REPO_TOKEN is not set, and the transcripts/reference "
            "folders aren't present locally — can't build the knowledge base."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            resp = httpx.get(
                f"https://api.github.com/repos/{CONTENT_REPO}/tarball/main",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                follow_redirects=True,
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            # Never surface the raw error — the request carries the token.
            raise RuntimeError(
                "Could not download the private content repo. Check that "
                "CONTENT_REPO_TOKEN is valid and has access to it."
            ) from None

        archive_path = tmp_path / "content.tar.gz"
        archive_path.write_bytes(resp.content)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp_path)

        # GitHub tarballs extract into one top-level dir, e.g. "owner-repo-<sha>"
        extracted_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        if not extracted_dirs:
            raise RuntimeError("Downloaded content repo archive was empty.")
        content_root = extracted_dirs[0]

        if not TRANSCRIPTS_DIR.exists():
            shutil.copytree(content_root / "transcripts", TRANSCRIPTS_DIR)
        if not REFERENCE_DIR.exists():
            shutil.copytree(content_root / "reference", REFERENCE_DIR)


def ensure_chroma_db_downloaded():
    """Downloads the prebuilt chroma_db from a GitHub Release in the private
    content repo, if one exists — this saves a fresh boot from having to
    re-embed every transcript from scratch, which is CPU-heavy enough to get
    the app throttled on Streamlit Cloud's free tier. Silently does nothing
    if unavailable/unconfigured; load_resources() then builds it from
    scratch as before."""
    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        return
    token = _get_secret("CONTENT_REPO_TOKEN")
    if not token:
        return
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        release_resp = httpx.get(
            f"https://api.github.com/repos/{CONTENT_REPO}/releases/latest",
            headers=headers,
            timeout=30,
        )
        release_resp.raise_for_status()
        asset = next(
            (a for a in release_resp.json().get("assets", []) if a["name"] == "chroma_db.tar.gz"),
            None,
        )
        if not asset:
            return
        asset_resp = httpx.get(
            asset["url"],
            headers={**headers, "Accept": "application/octet-stream"},
            follow_redirects=True,
            timeout=180,
        )
        asset_resp.raise_for_status()
    except httpx.HTTPError:
        return  # fall back to building from scratch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "chroma_db.tar.gz"
        archive_path.write_bytes(asset_resp.content)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp_path)
        extracted = tmp_path / "chroma_db"
        if extracted.exists():
            shutil.copytree(extracted, CHROMA_DB_DIR, dirs_exist_ok=True)


# ── Resource loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    voyage_api_key = _get_secret("VOYAGE_API_KEY")
    ensure_chroma_db_downloaded()

    def _open_collection():
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        coll = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return client, coll

    try:
        chroma_client, collection = _open_collection()
        needs_build = collection.count() == 0
    except Exception:
        # A downloaded DB that doesn't load cleanly (e.g. version mismatch) —
        # wipe it and rebuild from scratch instead of crashing.
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
        chroma_client, collection = _open_collection()
        needs_build = True

    if needs_build:
        # First boot with no usable chroma_db: download the private content
        # repo if needed, then build the index from it. Runs once — cached
        # by st.cache_resource.
        with st.spinner("Building knowledge base for the first time — this can take a few minutes..."):
            ensure_content_downloaded()
            ingest_from_dir(TRANSCRIPTS_DIR, voyage_api_key, collection, source_type="class", log=lambda *a: None)
            ingest_from_dir(REFERENCE_DIR, voyage_api_key, collection, source_type="reference", log=lambda *a: None)
    anthropic_api_key = _get_secret("ANTHROPIC_API_KEY")
    # Without an explicit timeout, a network hiccup can leave the app hanging
    # indefinitely with no visible error — fail fast instead.
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key, timeout=60.0)
    return voyage_api_key, collection, anthropic_client


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve(query: str, voyage_api_key: str, collection) -> list:
    query_embedding = embed_texts([query], input_type="query", api_key=voyage_api_key)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def _pretty_source_name(source_file: str) -> str:
    """Cleans a source filename for display: strips a trailing Vimeo id like
    '[797639202]' and underscores, without touching letter case — filenames
    already carry the correct casing (e.g. "LvR"), and .title() would break it."""
    name = source_file.replace("_", " ")
    name = re.sub(r"\s*\[\d+\]\s*$", "", name)
    return name.strip()


def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = _pretty_source_name(c["metadata"]["source_file"])
        ts = c["metadata"]["timestamp_label"]
        parts.append(f"[Excerpt {i} — {source}, {ts}]\n{c['text']}")
    return "\n\n".join(parts)


# The only session names ever shown to a student, regardless of which raw
# transcript (old cohort names, internal Q&A labels, co-host mentions, etc.)
# actually matched. Each topic is a set of words that must ALL appear
# (in any order, anywhere in the text) — not a fixed phrase, so "role as an
# artist" and "artist role" both match, and "a label" matches "labels".
CANONICAL_SESSIONS = [
    (["business", "side"], "Session 1: The Business Side w/ Leticia van Riel"),
    (["artist", "role"], "Session 2: The Artist Role w/ Victor Ruiz"),
    (["social", "media"], "Session 3: Social Media & Branding w/ Marcus O'Sullivan"),
    (["branding"], "Session 3: Social Media & Branding w/ Marcus O'Sullivan"),
    (["agent"], "Session 4: The Agent w/ Dylan First"),
    (["promoter"], "Session 5: Promoter w/ Victor De La Serna"),
    (["label"], "Session 6: Labels & Releases w/ Nick Garcia"),
    (["release"], "Session 6: Labels & Releases w/ Nick Garcia"),
    (["ads", "expert"], "Bonus Session: Becoming an Ads Expert"),
]
DEFAULT_SESSION = "Session 1: The Business Side w/ Leticia van Riel"


def _canonical_session_name(text: str):
    lowered = text.lower()
    for keywords, name in CANONICAL_SESSIONS:
        if all(k in lowered for k in keywords):
            return name
    return None


def suggest_class_to_watch(query: str, voyage_api_key: str, collection):
    """Finds the official session name to recommend for this query — raw
    transcript filenames (old cohort names, internal Q&A labels, co-host
    names) are never shown to students. Checks the query's own wording
    first (the most direct, reliable signal), then falls back to a majority
    vote across the top 10 retrieved chunks, then to a default session, so
    every answer gets a valid suggestion. Returns None only if the
    knowledge base has no class recordings at all."""
    query_name = _canonical_session_name(query)
    if query_name:
        return query_name

    query_embedding = embed_texts([query], input_type="query", api_key=voyage_api_key)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        where={"source_type": "class"},
        include=["metadatas"],
    )
    metadatas = results.get("metadatas", [[]])[0]
    if not metadatas:
        return None

    # A single top chunk is noisy (duplicate/near-duplicate segments of the
    # same file, or a tangentially-similar but off-topic file, often rank
    # highest) — a majority vote across the top 10 is far more reliable.
    votes = {}
    for meta in metadatas:
        name = _canonical_session_name(meta["source_file"])
        if name:
            votes[name] = votes.get(name, 0) + 1
    return max(votes, key=votes.get) if votes else DEFAULT_SESSION


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="LvR GPT",
        page_icon="⚡",
        layout="centered",
    )

    cookie_manager = get_cookie_manager()
    try_auto_login(cookie_manager)

    if not st.session_state.get("authenticated", False):
        show_login()
        return

    # Inject brand CSS
    st.markdown(LVR_CSS, unsafe_allow_html=True)

    try:
        voyage_api_key, collection, anthropic_client = load_resources()
    except Exception as e:
        st.error("Could not load the knowledge base.")
        st.exception(e)
        return

    # Chat history state: multiple threads per browser session
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0

    def new_chat():
        st.session_state.chat_counter += 1
        chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
        st.session_state.active_chat_id = chat_id

    if not st.session_state.chats:
        new_chat()

    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    # Sidebar
    with st.sidebar:
        st.markdown('<div style="padding: 1.25rem 0 0.5rem 0;">', unsafe_allow_html=True)
        with st.container(key="sticker-sidebar"):
            st.image(str(STICKER_PATH))
        st.markdown(
            """
            <div class="sidebar-wordmark">LvR GPT</div>
            <div class="sidebar-tag">LvR Mentoring</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin: 1.25rem 0 0.75rem;'></div>", unsafe_allow_html=True)

        with st.container(key="new-chat"):
            if st.button("+ New chat", use_container_width=True):
                new_chat()
                st.rerun()

        st.markdown("<div style='margin: 1rem 0 0.5rem;'></div>", unsafe_allow_html=True)

        with st.container(key="chat-list"):
            for chat_id in reversed(list(st.session_state.chats.keys())):
                chat = st.session_state.chats[chat_id]
                is_active = chat_id == st.session_state.active_chat_id
                label = ("● " if is_active else "") + chat["title"]
                if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()

    # Header — quiet by design; the sidebar already carries the wordmark and sticker
    st.markdown(
        """
        <div style="margin-bottom:1.75rem;">
            <div style="font-size:0.85rem;color:var(--lvr-text-muted);">
                Ask anything · Powered by LvR Mentoring lessons archive
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render existing messages
    for msg in active_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the classes..."):
        active_chat["messages"].append({"role": "user", "content": prompt})
        if active_chat["title"] == "New chat":
            active_chat["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")
        with st.chat_message("user"):
            st.markdown(prompt)

        print(f"[chat] retrieving context for: {prompt[:80]!r}", flush=True)
        chunks = retrieve(prompt, voyage_api_key, collection)
        context = build_context(chunks)
        print(f"[chat] retrieved {len(chunks)} chunks, calling Claude", flush=True)

        claude_messages = []
        for m in active_chat["messages"][:-1]:
            claude_messages.append({"role": m["role"], "content": m["content"]})

        user_message_with_context = (
            f"{prompt}\n\n---\nRelevant class transcript excerpts:\n\n{context}"
        )
        claude_messages.append({"role": "user", "content": user_message_with_context})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                with anthropic_client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=claude_messages,
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
                print("[chat] Claude response complete", flush=True)
            except Exception as e:
                print(f"[chat] Claude call failed: {e!r}", flush=True)
                response_placeholder.error(
                    "Something went wrong generating a response. Please try again."
                )
                st.exception(e)
                active_chat["messages"].pop()  # drop the user turn that never got a reply
                st.stop()

            suggestion = suggest_class_to_watch(prompt, voyage_api_key, collection)
            if suggestion:
                full_response += f"\n\n---\n🎬 **Suggested class to watch:** {suggestion}"

            response_placeholder.markdown(full_response)

        active_chat["messages"].append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
