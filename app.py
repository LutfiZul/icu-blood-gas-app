"""theme.py - tema klinikal yang flat dan ringkas untuk CDSS dashboard.

Guna:
    from theme import *
    inject_theme()
"""
import streamlit as st

INK = "#1B2530"
MUTED = "#5F6B7A"
LINE = "#D5DAE1"
BG = "#F4F5F7"
PANEL = "#FFFFFF"
ACCENT = "#1F4E79"
OK = "#2E7D5B"
WARN = "#B26A00"
BAD = "#B3372F"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#1B2530;--muted:#5F6B7A;--line:#D5DAE1;--bg:#F4F5F7;--panel:#FFFFFF;
--accent:#1F4E79;--ok:#2E7D5B;--warn:#B26A00;--bad:#B3372F;}
html,body,[class*="css"],.stApp{font-family:'IBM Plex Sans',sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.main .block-container{padding:1.5rem 2rem 3rem;max-width:1400px;}

section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] h2{font-size:12px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);margin:1.2rem 0 .4rem;}
.stButton>button{border-radius:3px;border:1px solid var(--line);background:var(--panel);
  color:var(--ink);font-weight:500;box-shadow:none;}
.stButton>button:hover{border-color:var(--accent);color:var(--accent);}
.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
.stTextInput input,.stNumberInput input{border-radius:3px;}

.pg-head{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:2px solid var(--ink);padding-bottom:8px;margin-bottom:18px;}
.pg-title{font-size:20px;font-weight:600;margin:0;}
.pg-sub{font-size:12px;color:var(--muted);}

.sec{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:6px;margin:26px 0 12px;}

.pt-bar{display:flex;flex-wrap:wrap;gap:28px;background:var(--panel);border:1px solid var(--line);
  padding:10px 16px;margin-bottom:16px;}
.pt-bar div span{display:block;font-size:11px;color:var(--muted);}
.pt-bar div b{font-family:'IBM Plex Mono',monospace;font-weight:500;font-size:14px;}

.stat{background:var(--panel);border:1px solid var(--line);padding:12px 14px;height:100%;}
.stat-l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.stat-v{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:500;margin-top:4px;
  font-variant-numeric:tabular-nums;}
.stat-u{font-size:12px;color:var(--muted);margin-left:4px;}
.stat-d{font-size:11px;color:var(--muted);margin-top:2px;font-family:'IBM Plex Mono',monospace;}
.stat.flag .stat-v{color:var(--bad);}
.stat.sm .stat-v{font-size:18px;}

.strip{display:flex;align-items:center;gap:10px;background:var(--panel);
  border:1px solid var(--line);padding:10px 14px;font-size:13px;margin-top:14px;}
.dot{width:9px;height:9px;border-radius:50%;flex:none;}
.dot.ok{background:var(--ok);}.dot.warn{background:var(--warn);}

.note{font-size:12px;color:var(--muted);line-height:1.5;margin-top:6px;}
.note b{color:var(--ink);font-weight:500;}
.panel-t{font-size:13px;font-weight:600;margin-bottom:2px;}

.stTable table{font-size:13px;border:1px solid var(--line);border-collapse:collapse;}
.stTable th{background:#EEF0F3;color:var(--ink);font-weight:600;border-bottom:1px solid var(--line);}
.stTable td{border-bottom:1px solid var(--line);}
.foot{font-size:11px;color:var(--muted);border-top:1px solid var(--line);
  margin-top:32px;padding-top:10px;}
"""


def inject_theme():
  st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_header(title, subtitle=""):
  st.markdown(
      f'<div class="pg-head"><h1 class="pg-title">{title}</h1>'
      f'<div class="pg-sub">{subtitle}</div></div>',
      unsafe_allow_html=True,
  )


def section(title):
  st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def patient_bar(items):
  """items: list of (label, value)."""
  cells = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in items)
  st.markdown(f'<div class="pt-bar">{cells}</div>', unsafe_allow_html=True)


def stat(label, value, unit="", delta="", flag=False, small=False):
  cls = "stat" + (" flag" if flag else "") + (" sm" if small else "")
  u = f'<span class="stat-u">{unit}</span>' if unit else ""
  d = f'<div class="stat-d">{delta}</div>' if delta else ""
  st.markdown(
      f'<div class="{cls}"><div class="stat-l">{label}</div>'
      f'<div class="stat-v">{value}{u}</div>{d}</div>',
      unsafe_allow_html=True,
  )


def status_strip(critical_hours):
  if not critical_hours:
    html = ('<span class="dot ok"></span><span><b>Stabil.</b> Tiada pengambilan'
            ' darah rutin diperlukan dalam tempoh 24 jam yang diramal.</span>')
  else:
    hrs = ", ".join(f"{h}j" for h in critical_hours)
    html = (f'<span class="dot warn"></span><span><b>Sampel disyorkan pada:'
            f'</b> {hrs}</span>')
  st.markdown(f'<div class="strip">{html}</div>', unsafe_allow_html=True)


def note(text):
  st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320):
  fig.update_layout(
      height=height,
      margin=dict(l=8, r=8, t=24, b=8),
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      font=dict(family="IBM Plex Sans", size=11, color=INK),
      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
      hovermode="x unified",
  )
  fig.update_xaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
  fig.update_yaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
  return fig
