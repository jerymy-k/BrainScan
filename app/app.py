import base64
import random
import tempfile
import time
from html import escape
from pathlib import Path

import streamlit as st

from brainscan.predict import (
    CLASS_NAMES,
    MODEL_PATH,
    VALID_EXTENSIONS,
    InvalidImageError,
    load_model,
    predict_image,
)

DATA_DIR = Path(__file__).resolve().parents[1] / 'Data'
LOW_CONFIDENCE = 0.60

CLASS_INFO = {
    'glioma': {
        'title': 'Glioma',
        'tumor': True,
        'text': 'A tumour that grows from glial cells, the supporting cells of the brain and spinal cord.',
    },
    'meningioma': {
        'title': 'Meningioma',
        'tumor': True,
        'text': 'A tumour that forms in the meninges, the membranes surrounding the brain and spinal cord.',
    },
    'notumor': {
        'title': 'No tumour',
        'tumor': False,
        'text': 'No tumour pattern was recognised in this scan.',
    },
    'pituitary': {
        'title': 'Pituitary tumour',
        'tumor': True,
        'text': 'A growth in the pituitary gland, located at the base of the brain behind the nose.',
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #070B14;
  --surface: rgba(17, 25, 43, 0.72);
  --surface-solid: #0F1626;
  --border: rgba(148, 163, 184, 0.14);
  --border-strong: rgba(45, 212, 191, 0.35);
  --text: #E2E8F0;
  --text-2: #94A3B8;
  --text-3: #64748B;
  --accent: #2DD4BF;
  --accent-2: #38BDF8;
  --track: #1E293B;
  --bar-muted: #475569;
  --good: #34D399;
  --warn: #FBBF24;
  --critical: #F87171;
}

[data-testid="stHtml"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(900px 500px at 12% -10%, rgba(45, 212, 191, 0.10), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(56, 189, 248, 0.08), transparent 55%),
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }

/* ---------- hero ---------- */
.bs-hero { display: flex; align-items: center; gap: 18px; margin-bottom: 6px; }
.bs-logo {
  width: 56px; height: 56px; border-radius: 16px; flex: none;
  display: grid; place-items: center;
  background: linear-gradient(135deg, rgba(45,212,191,.18), rgba(56,189,248,.12));
  border: 1px solid var(--border-strong);
  box-shadow: 0 0 32px rgba(45, 212, 191, 0.18);
}
.bs-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 2.6rem; line-height: 1; letter-spacing: -0.03em; margin: 0;
  background: linear-gradient(90deg, #F8FAFC 0%, #A5F3FC 60%, var(--accent) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.bs-sub { color: var(--text-2); font-size: 1.02rem; margin-top: 6px; }
.bs-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 28px; }
.bs-chip {
  font-size: .78rem; font-weight: 500; color: var(--text-2);
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--border); background: rgba(15, 22, 38, .6);
}
.bs-chip b { color: var(--text); font-weight: 600; }

/* ---------- cards ---------- */
[class*="st-key-bs-card"] {
  background: var(--surface);
  border: 1px solid var(--border) !important;
  border-radius: 20px;
  padding: 22px 22px 18px;
  backdrop-filter: blur(14px);
  box-shadow: 0 20px 50px rgba(0, 0, 0, .35);
}
.bs-eyebrow {
  font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 4px;
}
.bs-h2 { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; color: var(--text); margin: 0 0 14px; }

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(7, 11, 20, .55);
  border: 1.5px dashed rgba(45, 212, 191, .35);
  border-radius: 16px; padding: 28px 18px;
  transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: rgba(45, 212, 191, .05); }
[data-testid="stFileUploaderDropzone"] button {
  border-radius: 10px; border: 1px solid var(--border-strong);
  background: rgba(45, 212, 191, .1); color: var(--accent);
}

/* ---------- scan preview ---------- */
.bs-scan {
  position: relative; border-radius: 16px; overflow: hidden;
  border: 1px solid var(--border); background: #000; aspect-ratio: 1 / 1;
}
.bs-scan img { width: 100%; height: 100%; object-fit: contain; display: block; }
.bs-scan::after {
  content: ""; position: absolute; left: 0; right: 0; height: 30%;
  background: linear-gradient(180deg, transparent, rgba(45, 212, 191, .22), transparent);
  border-bottom: 1px solid rgba(45, 212, 191, .6);
  animation: bs-sweep 2.8s ease-in-out 1 forwards;
  pointer-events: none;
}
@keyframes bs-sweep { 0% { top: -30%; opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
.bs-corner { position: absolute; width: 18px; height: 18px; border-color: var(--accent); border-style: solid; opacity: .8; }
.bs-corner.tl { top: 10px; left: 10px; border-width: 2px 0 0 2px; border-radius: 4px 0 0 0; }
.bs-corner.tr { top: 10px; right: 10px; border-width: 2px 2px 0 0; border-radius: 0 4px 0 0; }
.bs-corner.bl { bottom: 10px; left: 10px; border-width: 0 0 2px 2px; border-radius: 0 0 0 4px; }
.bs-corner.br { bottom: 10px; right: 10px; border-width: 0 2px 2px 0; border-radius: 0 0 4px 0; }
.bs-filename { color: var(--text-3); font-size: .78rem; margin-top: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- verdict ---------- */
.bs-verdict { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.bs-status {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .78rem; font-weight: 600; padding: 5px 10px; border-radius: 999px; margin-bottom: 10px;
}
.bs-status.good { color: var(--good); background: rgba(52, 211, 153, .1); border: 1px solid rgba(52, 211, 153, .3); }
.bs-status.warn { color: var(--warn); background: rgba(251, 191, 36, .1); border: 1px solid rgba(251, 191, 36, .3); }
.bs-label { font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.05; color: var(--text); }
.bs-desc { color: var(--text-2); font-size: .92rem; margin-top: 8px; max-width: 420px; }
.bs-conf { text-align: right; flex: none; }
.bs-conf-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; color: var(--accent); line-height: 1; font-variant-numeric: tabular-nums; }
.bs-conf-label { color: var(--text-3); font-size: .74rem; text-transform: uppercase; letter-spacing: .12em; margin-top: 6px; }

.bs-divider { height: 1px; background: var(--border); margin: 22px 0 18px; }

/* ---------- probability bars ---------- */
.bs-bars { display: flex; flex-direction: column; gap: 14px; }
.bs-row { display: grid; grid-template-columns: 130px 1fr 64px; align-items: center; gap: 14px; }
.bs-row-name { color: var(--text-2); font-size: .9rem; font-weight: 500; }
.bs-row.top .bs-row-name { color: var(--text); font-weight: 600; }
.bs-track { height: 10px; border-radius: 999px; background: var(--track); overflow: hidden; }
.bs-fill {
  height: 100%; border-radius: 999px; background: var(--bar-muted);
  transform-origin: left; animation: bs-grow .9s cubic-bezier(.2, .8, .2, 1) both;
}
.bs-row.top .bs-fill { background: linear-gradient(90deg, var(--accent-2), var(--accent)); box-shadow: 0 0 14px rgba(45, 212, 191, .45); }
@keyframes bs-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.bs-row-value { text-align: right; color: var(--text-2); font-size: .88rem; font-variant-numeric: tabular-nums; }
.bs-row.top .bs-row-value { color: var(--text); font-weight: 600; }
.bs-row:hover .bs-row-name, .bs-row:hover .bs-row-value { color: var(--text); }
.bs-meta {
  display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 22px; padding-top: 14px;
  border-top: 1px solid var(--border); color: var(--text-3); font-size: .78rem;
}
.bs-meta b { color: var(--text-2); font-weight: 600; font-variant-numeric: tabular-nums; }

/* ---------- empty state ---------- */
.bs-empty { min-height: 420px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 14px; color: var(--text-3); }
.bs-empty-ring {
  width: 120px; height: 120px; border-radius: 50%;
  display: grid; place-items: center;
  border: 1px dashed rgba(45, 212, 191, .3);
  animation: bs-pulse 3s ease-in-out infinite;
}
@keyframes bs-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(45, 212, 191, .18); } 50% { box-shadow: 0 0 0 18px rgba(45, 212, 191, 0); } }
.bs-empty h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text); font-size: 1.15rem; font-weight: 600; margin: 6px 0 0; }
.bs-empty p { max-width: 320px; font-size: .9rem; margin: 0; }

/* ---------- notes ---------- */
.bs-note {
  display: flex; gap: 10px; align-items: flex-start;
  font-size: .85rem; color: var(--text-2); line-height: 1.5;
  padding: 12px 14px; border-radius: 12px; margin-top: 18px;
  background: rgba(15, 22, 38, .7); border: 1px solid var(--border);
}
.bs-note.warn { border-color: rgba(251, 191, 36, .3); color: #FDE68A; }
.bs-note.error { border-color: rgba(248, 113, 113, .35); color: #FECACA; }
.bs-footer { color: var(--text-3); font-size: .8rem; text-align: center; margin-top: 36px; }

/* ---------- pills ---------- */
[data-testid="stPills"] button, [data-testid="stButtonGroup"] button { border-radius: 999px !important; }

@media (max-width: 640px) {
  .bs-title { font-size: 2rem; }
  .bs-verdict { flex-direction: column; }
  .bs-conf { text-align: left; }
  .bs-row { grid-template-columns: 96px 1fr 52px; gap: 10px; }
}
</style>
"""

ICON_PATHS = {
    'brain': '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/>'
             '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/>',
    'check': '<path d="M20 6 9 17l-5-5"/>',
    'alert': '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
    'info': '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
}
NOTE_COLORS = {'': '#94A3B8', 'warn': '#FBBF24', 'error': '#F87171'}


def icon(name, size, color, stroke=2.0, style=''):
    # st.html strips inline <svg>, so icons are embedded as SVG data-URI images
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[name]}</svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" alt="" style="flex:none;{style}">'


@st.cache_resource(show_spinner='Loading the model…')
def get_model():
    return load_model()


def card(name):
    # The key adds a "st-key-bs-card-<name>" class that the CSS styles as a card
    return st.container(key=f'bs-card-{name}')


def note(text, kind=''):
    st.html(f'<div class="bs-note {kind}">{icon("info", 16, NOTE_COLORS[kind], style="margin-top:2px")}<div>{text}</div></div>')


def render_hero():
    st.html(f"""
    <div class="bs-hero">
      <div class="bs-logo">{icon('brain', 30, '#2DD4BF', 1.6)}</div>
      <div>
        <h1 class="bs-title">BrainScan</h1>
        <div class="bs-sub">AI-assisted brain tumour classification from MRI scans</div>
      </div>
    </div>
    <div class="bs-chips">
      <span class="bs-chip"><b>4</b> classes</span>
      <span class="bs-chip"><b>97.2%</b> test accuracy</span>
      <span class="bs-chip"><b>7,023</b> MRI scans</span>
      <span class="bs-chip">Custom <b>CNN</b> · 224×224</span>
    </div>
    """)


def render_scan(image_bytes, suffix, filename):
    mime = 'jpeg' if suffix in ('.jpg', '.jpeg') else suffix.lstrip('.')
    b64 = base64.b64encode(image_bytes).decode()
    st.html(f"""
    <div class="bs-scan">
      <img src="data:image/{mime};base64,{b64}" alt="Uploaded MRI scan">
      <span class="bs-corner tl"></span><span class="bs-corner tr"></span>
      <span class="bs-corner bl"></span><span class="bs-corner br"></span>
    </div>
    <div class="bs-filename">{escape(filename)}</div>
    """)


def render_empty():
    st.html(f"""
    <div class="bs-empty">
      <div class="bs-empty-ring">{icon('brain', 46, '#2DD4BF', 1.4)}</div>
      <h3>Awaiting a scan</h3>
      <p>Upload a brain MRI image, or pick a sample scan, to see the model's diagnosis and class probabilities.</p>
    </div>
    """)


def render_result(result, elapsed_ms):
    label = result['label']
    info = CLASS_INFO[label]
    status = (
        f'<span class="bs-status warn">{icon("alert", 13, "#FBBF24", 2.4)} Tumour detected</span>' if info['tumor']
        else f'<span class="bs-status good">{icon("check", 13, "#34D399", 2.6)} No tumour detected</span>'
    )
    probs = sorted(result['probabilities'].items(), key=lambda kv: kv[1], reverse=True)
    rows = ''.join(
        f"""
        <div class="bs-row{' top' if name == label else ''}" title="{CLASS_INFO[name]['title']}: {p * 100:.2f}%">
          <span class="bs-row-name">{CLASS_INFO[name]['title']}</span>
          <div class="bs-track"><div class="bs-fill" style="width:{max(p * 100, 0.8):.2f}%;animation-delay:{i * 90}ms"></div></div>
          <span class="bs-row-value">{p * 100:.1f}%</span>
        </div>"""
        for i, (name, p) in enumerate(probs)
    )
    st.html(f"""
    <div class="bs-eyebrow">Diagnosis</div>
    <div class="bs-verdict">
      <div>
        {status}
        <div class="bs-label">{info['title']}</div>
        <div class="bs-desc">{info['text']}</div>
      </div>
      <div class="bs-conf">
        <div class="bs-conf-value">{result['confidence'] * 100:.1f}<span style="font-size:1.3rem">%</span></div>
        <div class="bs-conf-label">Confidence</div>
      </div>
    </div>
    <div class="bs-divider"></div>
    <div class="bs-eyebrow">Class probabilities</div>
    <div class="bs-bars" style="margin-top:12px">{rows}</div>
    <div class="bs-meta">
      <span>Analysed in <b>{elapsed_ms:.0f} ms</b></span>
      <span>Input <b>224×224</b></span>
      <span>Model <b>CNN · 4.8M params</b></span>
    </div>
    """)
    if result['confidence'] < LOW_CONFIDENCE:
        note(
            f"<b>Low confidence.</b> The model is less than {LOW_CONFIDENCE:.0%} sure about this scan. "
            "Treat the result with caution and consider a second opinion.",
            'warn',
        )


def sample_files():
    if not DATA_DIR.is_dir():
        return {}
    return {
        name: sorted(f for f in (DATA_DIR / name).glob('Te-*') if f.suffix.lower() in VALID_EXTENSIONS)
        for name in CLASS_NAMES
        if (DATA_DIR / name).is_dir()
    }


def pick_sample():
    choice = st.session_state.get('sample_choice')
    st.session_state.sample_path = None
    if choice:
        name = next(n for n, i in CLASS_INFO.items() if i['title'] == choice)
        files = sample_files().get(name)
        if files:
            st.session_state.sample_path = random.choice(files)


def run_prediction(image_bytes, suffix, model):
    # predict_image works on paths, so the upload goes through a temp file:
    # this reuses exactly the same validation and preprocessing as everywhere else.
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        return predict_image(tmp.name, model=model)


def main():
    st.set_page_config(page_title='BrainScan · MRI Tumour Classifier', page_icon='🧠', layout='wide')
    st.html(CSS)
    render_hero()

    try:
        model = get_model()
    except (OSError, ValueError):
        note(
            f"<b>Model not found.</b> Expected a trained model at <code>{escape(str(MODEL_PATH))}</code>. "
            "Run the training notebook first to create it.",
            'error',
        )
        st.stop()

    left, right = st.columns([5, 7], gap='large')

    with left, card('upload'):
        st.html('<div class="bs-eyebrow">Step 1</div><div class="bs-h2">Upload an MRI scan</div>')
        uploaded = st.file_uploader(
            'MRI image',
            type=[ext.lstrip('.') for ext in sorted(VALID_EXTENSIONS)],
            label_visibility='collapsed',
        )

        samples = sample_files()
        if samples and not uploaded:
            st.pills(
                'Or try a sample scan',
                [CLASS_INFO[n]['title'] for n in samples],
                key='sample_choice',
                on_change=pick_sample,
            )

        image_bytes = suffix = filename = None
        if uploaded:
            image_bytes, suffix, filename = uploaded.getvalue(), Path(uploaded.name).suffix.lower(), uploaded.name
        elif st.session_state.get('sample_path'):
            path = st.session_state.sample_path
            image_bytes, suffix, filename = path.read_bytes(), path.suffix.lower(), f'Sample · {path.name}'

        if image_bytes:
            render_scan(image_bytes, suffix, filename)

    with right, card('result'):
        if not image_bytes:
            render_empty()
        else:
            try:
                with st.spinner('Analysing scan…'):
                    start = time.perf_counter()
                    result = run_prediction(image_bytes, suffix, model)
                    elapsed_ms = (time.perf_counter() - start) * 1000
            except InvalidImageError as e:
                note(f'<b>This file can’t be analysed.</b> {escape(str(e).split(":")[0])}. Please upload a valid MRI image.', 'error')
            except Exception:
                note('<b>Something went wrong</b> while analysing this scan. Please try another image.', 'error')
            else:
                render_result(result, elapsed_ms)

    note(
        '<b>For research and education only.</b> BrainScan is a student project and not a medical device. '
        'Its output must never replace diagnosis by a qualified radiologist.'
    )
    st.html('<div class="bs-footer">BrainScan · CNN trained on brain MRI scans · glioma · meningioma · pituitary · no tumour</div>')


main()
