"""
ui.py
=====
Design system untuk antarmuka Streamlit: tema gelap "Midnight Crimson",
hero, stepper pipeline, kartu hasil dengan score bar, badge verifikasi sitasi.
Semua fungsi mengembalikan string HTML (dipakai dengan unsafe_allow_html=True).
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Sequence, Tuple

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --bg:#07090f; --s1:#0e1220; --s2:#151a2c; --bd:#232a40; --tx:#F5F7FC; --mut:#C3CAE0;
  --ac:#E5484D; --ac2:#F5B971; --ok:#3DD68C; --warn:#F5B971; --bad:#FF6369;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; color:var(--tx); }
.stApp{
  background:
    radial-gradient(1100px 480px at 8% -10%, rgba(229,72,77,.16), transparent 60%),
    radial-gradient(900px 420px at 100% 0%, rgba(245,185,113,.08), transparent 60%),
    var(--bg);
}
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.2rem; max-width:1260px; }
h1,h2,h3{ letter-spacing:-.01em; }
@keyframes fadeUp{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;} }
@keyframes pulse{ 0%,100%{box-shadow:0 0 0 0 rgba(61,214,140,.55);} 50%{box-shadow:0 0 0 6px rgba(61,214,140,0);} }

/* ---------- hero ---------- */
.hero{ position:relative; overflow:hidden; border:1px solid var(--bd); border-radius:20px;
  padding:1.7rem 2rem 1.5rem; margin-bottom:1rem; animation:fadeUp .5s ease both;
  background:linear-gradient(135deg, rgba(139,0,0,.55) 0%, rgba(21,26,44,.9) 55%, rgba(14,18,32,.95) 100%); }
.hero:before{ content:""; position:absolute; inset:0; opacity:.35; pointer-events:none;
  background-image:linear-gradient(var(--bd) 1px, transparent 1px), linear-gradient(90deg, var(--bd) 1px, transparent 1px);
  background-size:36px 36px; mask-image:radial-gradient(circle at 80% 20%, #000, transparent 70%); }
.hero > * { position:relative; }
.eyebrow{ display:inline-flex; gap:.5rem; align-items:center; font-size:.72rem; font-weight:600; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ac2); background:rgba(245,185,113,.1); border:1px solid rgba(245,185,113,.25);
  padding:.25rem .7rem; border-radius:99px; }
.hero h1{ font-family:'Fraunces',serif; font-size:2.25rem; line-height:1.1; margin:.8rem 0 .4rem; color:#fff; }
.hero h1 em{ font-style:normal; background:linear-gradient(90deg,#FF7A7F,#F5B971); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.hero p{ color:#E1E6F5; max-width:720px; margin:0 0 1rem; font-size:1rem; }
.stats{ display:flex; gap:.7rem; flex-wrap:wrap; }
.stat{ background:rgba(7,9,15,.55); border:1px solid var(--bd); border-radius:12px; padding:.55rem .95rem; min-width:112px; }
.stat b{ display:block; font-size:1.25rem; color:#fff; font-variant-numeric:tabular-nums; }
.stat span{ font-size:.72rem; color:var(--mut); text-transform:uppercase; letter-spacing:.06em; }

/* ---------- pipeline stepper ---------- */
.pipe{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 1.1rem; }
.step{ display:flex; align-items:center; gap:.55rem; background:var(--s1); border:1px solid var(--bd);
  border-radius:12px; padding:.45rem .8rem; font-size:.82rem; transition:.2s; }
.step:hover{ border-color:#3a4366; transform:translateY(-1px); }
.dot{ width:9px; height:9px; border-radius:50%; background:#6b7494; flex:none; }
.step.done .dot{ background:var(--ok); animation:pulse 2.2s infinite; }
.step.warn .dot{ background:var(--warn); }
.step small{ color:var(--mut); display:block; font-size:.7rem; }
.step b{ font-weight:600; }

/* ---------- result cards ---------- */
.rc{ background:linear-gradient(180deg, var(--s1), #0b0f1a); border:1px solid var(--bd); border-radius:16px;
  padding:1rem 1.15rem .95rem; margin:.75rem 0 .3rem; position:relative; animation:fadeUp .45s ease both;
  transition:border-color .2s, transform .2s, box-shadow .2s; }
.rc:hover{ border-color:rgba(229,72,77,.6); transform:translateY(-2px); box-shadow:0 10px 30px rgba(0,0,0,.35); }
.rc:before{ content:""; position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px;
  background:linear-gradient(var(--ac), var(--ac2)); }
.rc-top{ display:flex; gap:.8rem; align-items:flex-start; justify-content:space-between; }
.rank{ font-family:'JetBrains Mono',monospace; color:var(--ac2); font-size:.8rem; }
.rc-title{ font-weight:650; font-size:1.02rem; color:#fff; margin:.1rem 0 .25rem; }
.chips{ display:flex; gap:.35rem; flex-wrap:wrap; margin:.2rem 0 .5rem; }
.chip{ font-size:.72rem; padding:.12rem .6rem; border-radius:99px; background:var(--s2); border:1px solid var(--bd); color:#EEF1FA; }
.chip.ac{ background:rgba(229,72,77,.14); border-color:rgba(229,72,77,.4); color:#FFB3B6; }
.chip.ok{ background:rgba(61,214,140,.12); border-color:rgba(61,214,140,.4); color:#8CF0BE; }
.chip.warn{ background:rgba(245,185,113,.12); border-color:rgba(245,185,113,.4); color:#F7CC9A; }
.chip.bad{ background:rgba(255,99,105,.12); border-color:rgba(255,99,105,.4); color:#FF9DA1; }
.score{ min-width:120px; text-align:right; }
.score b{ font-family:'JetBrains Mono',monospace; font-size:.85rem; color:#fff; }
.bar{ height:6px; background:var(--s2); border-radius:99px; overflow:hidden; margin-top:.3rem; }
.bar > i{ display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,var(--ac),var(--ac2)); }
.snip{ font-size:.92rem; line-height:1.6; color:#E6EAF7; }
mark{ background:rgba(245,185,113,.28); color:#FFE2BD; padding:0 3px; border-radius:4px; }
.cite{ display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.72rem; font-weight:600; padding:0 .45rem;
  margin:0 .1rem; border-radius:6px; background:rgba(229,72,77,.18); border:1px solid rgba(229,72,77,.5); color:#FFB3B6; }

/* ---------- verifikasi ---------- */
.verdict{ display:flex; gap:.8rem; align-items:center; background:var(--s1); border:1px solid var(--bd);
  border-radius:14px; padding:.7rem 1rem; margin:.6rem 0; }
.ring{ --p:0; width:54px; height:54px; border-radius:50%; flex:none; display:grid; place-items:center;
  background:conic-gradient(var(--c,#3DD68C) calc(var(--p)*1%), var(--s2) 0); }
.ring span{ width:42px; height:42px; border-radius:50%; background:var(--s1); display:grid; place-items:center;
  font-size:.78rem; font-weight:700; color:#fff; }
.verdict p{ margin:0; font-size:.85rem; color:#E6EAF7; }
.verdict strong{ color:#fff; }

/* ---------- empty state ---------- */
.empty{ text-align:center; padding:2.2rem 1rem; border:1px dashed var(--bd); border-radius:16px; color:var(--mut); margin:.8rem 0; }
.empty .ic{ font-size:2rem; } .empty b{ display:block; color:#fff; margin:.3rem 0 .15rem; }
.brand{ font-family:'Fraunces',serif; font-size:1.35rem; margin-bottom:.4rem; color:#fff; }
.brand span{ color:var(--ac); }

/* ---------- komponen bawaan streamlit ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:.3rem; border-bottom:1px solid var(--bd); overflow-x:auto; }
.stTabs [data-baseweb="tab"]{ background:transparent; border-radius:10px 10px 0 0; padding:.55rem 1rem; color:var(--mut);
  font-weight:600; white-space:nowrap; }
.stTabs [aria-selected="true"]{ color:#fff !important; background:var(--s1) !important; border:1px solid var(--bd); border-bottom:none; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--ac) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input{ background:var(--s1) !important; border:1px solid var(--bd) !important;
  border-radius:12px !important; color:var(--tx) !important; }
.stTextInput input:focus, .stTextArea textarea:focus{ border-color:var(--ac) !important; box-shadow:0 0 0 3px rgba(229,72,77,.2) !important; }
div[data-testid="stTextInput"] input{ font-size:1.05rem; padding:.8rem 1rem; }
div.stButton > button, div.stDownloadButton > button{ background:var(--s2); color:#fff; border:1px solid var(--bd); border-radius:10px;
  font-weight:600; transition:.15s; }
div.stButton > button:hover, div.stDownloadButton > button:hover{ border-color:var(--ac); color:#fff; transform:translateY(-1px); }
div.stButton > button[kind="primary"]{ background:linear-gradient(135deg,#B3141A,var(--ac)); border:none; }
div[data-testid="stMetric"]{ background:var(--s1); border:1px solid var(--bd); border-radius:12px; padding:.6rem .9rem; }
section[data-testid="stSidebar"]{ background:#0a0d17; border-right:1px solid var(--bd); }
details{ border-radius:12px !important; border-color:var(--bd) !important; background:var(--s1); }
.stChatMessage{ background:var(--s1); border:1px solid var(--bd); border-radius:16px; }
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button{ border-radius:99px !important; }
.st-key-nav [data-testid="stSegmentedControl"] button{ padding:.55rem 1.05rem; font-weight:600; font-size:.95rem; }
.st-key-nav [data-testid="stSegmentedControl"]{ margin-bottom:.4rem; }
/* ---------- keterbacaan teks bawaan streamlit di background gelap ---------- */
:where(.stApp) :where(p, li, span, label, h1, h2, h3, h4, small, div),
:where([data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stCheckbox"], [data-testid="stRadio"],
  [data-testid="stToggle"], [data-testid="stSlider"], [data-testid="stSelectbox"], [data-testid="stExpander"] summary,
  [data-testid="stChatMessage"], section[data-testid="stSidebar"]) :where(*){ color:var(--tx); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stApp small,
[data-testid="stMetricLabel"] *, [data-testid="stTooltipIcon"]{ color:var(--mut) !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder{ color:#9AA3C0 !important; opacity:1; }
[data-testid="stChatInput"] textarea, [data-testid="stSelectbox"] div[data-baseweb="select"] *{ color:var(--tx) !important; }
[data-testid="stSegmentedControl"] button, [data-testid="stPills"] button{ color:#E6EAF7 !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stPills"] button[aria-pressed="true"]{ color:#fff !important; }
[data-testid="stMetricValue"] *{ color:#fff !important; }
a{ color:#FFB3B6; }
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"]{ background:#B3141A !important; border-color:#E5484D !important; }
@media (max-width:760px){ .hero h1{font-size:1.65rem;} .rc-top{flex-direction:column;} .score{text-align:left; width:100%;} }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def hero(stats: Sequence[Tuple[str, str]]) -> str:
    cells = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in stats)
    return (
        '<div class="hero">'
        '<span class="eyebrow">⚖️ UTS STKI · UDINUS · Hybrid Retrieval + RAG</span>'
        '<h1>Dari ribuan pasal menjadi <em>jawaban bersitasi</em>.</h1>'
        '<p>Mesin pencari peraturan perundang-undangan Indonesia: Boolean, TF-IDF, BM25, LSA, embedding, '
        'hybrid, dan asisten RAG dengan verifikasi sitasi otomatis. Komponen leksikal ditulis dari nol.</p>'
        f'<div class="stats">{cells}</div></div>'
    )


def pipeline(steps: Iterable[Tuple[str, str, str]]) -> str:
    """steps: (label, state['done'|'warn'|'idle'], detail)"""
    out = []
    for label, state, detail in steps:
        out.append(f'<div class="step {state}"><span class="dot"></span><div><b>{esc(label)}</b>'
                   f'<small>{esc(detail)}</small></div></div>')
    return f'<div class="pipe">{"".join(out)}</div>'


def result_card(rank: int, title: str, chips_html: str, snippet_html: str,
                score: float, ratio: float, score_label: str = "Skor") -> str:
    pct = max(4, min(100, int(round(ratio * 100))))
    delay = min(rank, 10) * 45
    return (
        f'<div class="rc" style="animation-delay:{delay}ms">'
        '<div class="rc-top"><div>'
        f'<span class="rank">#{rank:02d}</span>'
        f'<div class="rc-title">{esc(title)}</div>'
        f'<div class="chips">{chips_html}</div></div>'
        f'<div class="score"><b>{esc(score_label)} {score:.4f}</b>'
        f'<div class="bar"><i style="width:{pct}%"></i></div></div></div>'
        f'<div class="snip">{snippet_html}</div></div>'
    )


def empty(icon: str, title: str, text: str) -> str:
    return f'<div class="empty"><div class="ic">{icon}</div><b>{esc(title)}</b>{esc(text)}</div>'


def verdict(grounding: float | None, headline: str, detail: str) -> str:
    if grounding is None:
        pct, color = 0, "#8B93AD"
        label = "–"
    else:
        pct = int(round(grounding * 100))
        label = f"{pct}%"
        color = "#3DD68C" if pct >= 80 else ("#F5B971" if pct >= 50 else "#FF6369")
    return (
        f'<div class="verdict"><div class="ring" style="--p:{pct};--c:{color}"><span>{label}</span></div>'
        f'<p><strong>{esc(headline)}</strong><br>{esc(detail)}</p></div>'
    )


_CITE_RE = re.compile(r"\[S(\d+)\]")


def cite_markdown(text: str) -> str:
    """Ubah [S1] menjadi lencana Markdown bawaan Streamlit; format Markdown jawaban LLM tetap utuh."""
    return _CITE_RE.sub(lambda m: f":red-badge[S{m.group(1)}]", text)