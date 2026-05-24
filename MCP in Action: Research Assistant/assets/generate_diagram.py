"""
Architecture diagram — clean, precise, blog-ready.
Run:  python assets/generate_diagram.py
Out:  assets/architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = "#F8F9FC"
WHITE  = "#FFFFFF"
DARK   = "#1E1E2E"
GRAY   = "#6B7280"
LGRAY  = "#E5E7EB"
BLUE   = "#2563EB"
PURPLE = "#7C3AED"
TEAL   = "#0D9488"
GREEN  = "#059669"
AMBER  = "#B45309"
RED    = "#DC2626"

fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Drawing helpers ────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc, ec, lw=1.6, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def txt(x, y, s, sz=9, c=DARK, w="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=sz, color=c, fontweight=w,
            ha=ha, va=va, zorder=5)

def arrow_h(x1, x2, y, color, lbl=None, lbl_dy=0.18):
    """Horizontal arrow x1→x2 at height y."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.6, mutation_scale=12), zorder=6)
    if lbl:
        ax.text((x1+x2)/2, y+lbl_dy, lbl, fontsize=6.8, color=color,
                ha="center", style="italic", zorder=7,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.2))

def arrow_v(x, y1, y2, color, lbl=None, lbl_dx=0.18):
    """Vertical arrow y1→y2 at position x."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.6, mutation_scale=12), zorder=6)
    if lbl:
        ax.text(x+lbl_dx, (y1+y2)/2, lbl, fontsize=6.8, color=color,
                ha="left", style="italic", zorder=7,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.2))

def arrow_path(pts, color, lbl=None, lbl_pt=None):
    """Multi-segment path; last segment gets arrowhead."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.6, zorder=5,
            solid_capstyle="round")
    # arrowhead on the last segment
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.6, mutation_scale=12), zorder=6)
    if lbl and lbl_pt:
        ax.text(lbl_pt[0], lbl_pt[1], lbl, fontsize=6.8, color=color,
                ha="center", style="italic", zorder=7,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(9, 9.65, "Personal AI Research Assistant  —  Architecture",
    sz=15, w="bold")
txt(9, 9.30, "FastMCP  ·  LangChain ReAct Agent  ·  OpenAI GPT-4.1-mini  ·  DuckDuckGo",
    sz=9, c=GRAY)
ax.plot([0.3, 17.7], [9.08, 9.08], color=LGRAY, lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS  (tweak these to shift whole columns/rows)
# ══════════════════════════════════════════════════════════════════════════════
#  Columns          x-start    x-end
COL_USER_X  = 0.30 ;  COL_USER_W  = 1.90   # 0.30 – 2.20
COL_AGT_X   = 2.50 ;  COL_AGT_W   = 6.20   # 2.50 – 8.70
COL_MCP_X   = 8.90 ;  COL_MCP_W   = 5.80   # 8.90 – 14.70
COL_EXT_X   = 15.0 ;  COL_EXT_W   = 2.70   # 15.0 – 17.70

#  Vertical band
SEC_Y    = 0.40   # section bottom
SEC_H    = 8.50   # section height → top = 8.90

# Inner padding inside sections
PAD = 0.35

# ── Agent inner boxes ──────────────────────────────────────────────────────────
# Box coords:  (x, y, w, h)  — all within COL_AGT
IA_X  = COL_AGT_X + PAD        # 2.85
IA_W  = COL_AGT_W - 2*PAD      # 5.50

LLM_Y = 7.10 ; LLM_H = 1.45    # top = 8.55
MCP_Y = 5.45 ; MCP_H = 1.40    # top = 6.85
STP_Y = 4.00 ; STP_H = 1.15    # top = 5.15
LOP_Y = 0.70 ; LOP_H = 3.10    # top = 3.80

# ── MCP Server inner boxes ────────────────────────────────────────────────────
IM_X = COL_MCP_X + PAD         # 9.25
IM_W = COL_MCP_W - 2*PAD       # 5.10

TOOL_Y = 5.60 ; TOOL_H = 2.95  # top = 8.55
PRMT_Y = 3.20 ; PRMT_H = 2.10  # top = 5.30
RESC_Y = 0.70 ; RESC_H = 2.20  # top = 2.90

# ── External boxes ─────────────────────────────────────────────────────────────
OAI_Y  = 7.10 ; OAI_H  = 1.45  # OpenAI   top = 8.55
DDG_Y  = 5.30 ; DDG_H  = 1.45  # DDG      top = 6.75
DAT_Y  = 0.40 ; DAT_H  = 4.55  # DataStore top = 4.95

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — USER
# ══════════════════════════════════════════════════════════════════════════════
rbox(COL_USER_X, 3.80, COL_USER_W, 1.90, WHITE, BLUE, lw=2.0)
txt(COL_USER_X + COL_USER_W/2, 5.00, "User", sz=11, w="bold")
txt(COL_USER_X + COL_USER_W/2, 4.52, 'python -m agent\n.main_agent "..."', sz=7.5, c=GRAY)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — AGENT SECTION
# ══════════════════════════════════════════════════════════════════════════════
rbox(COL_AGT_X, SEC_Y, COL_AGT_W, SEC_H, "#F3F0FF", PURPLE, lw=2.0, z=1)
txt(COL_AGT_X + 0.18, SEC_Y + SEC_H - 0.28,
    "LangChain ReAct Agent", sz=9.5, w="bold", c=PURPLE, ha="left")

# 2a LLM
rbox(IA_X, LLM_Y, IA_W, LLM_H, WHITE, PURPLE)
cx = IA_X + IA_W/2
txt(cx, LLM_Y + LLM_H*0.65, "GPT-4.1-mini", sz=11, w="bold")
txt(cx, LLM_Y + LLM_H*0.28, "ChatOpenAI  ·  .bind_tools()  ·  temperature=0", sz=8, c=GRAY)

# 2b MCP Client
rbox(IA_X, MCP_Y, IA_W, MCP_H, WHITE, PURPLE)
txt(cx, MCP_Y + MCP_H*0.65, "MCP Client", sz=11, w="bold")
txt(cx, MCP_Y + MCP_H*0.28, "MultiServerMCPClient  ·  Streamable HTTP  ·  :8001/mcp", sz=8, c=GRAY)

# 2c Step pills
rbox(IA_X, STP_Y, IA_W, STP_H, WHITE, PURPLE, lw=1.4)
step_data = [
    (RED,    "1", "Resources",  "config://agent · notes://all",     IA_X + IA_W*0.17),
    (AMBER,  "2", "Prompts",    "research_prompt · summarize_prompt", IA_X + IA_W*0.50),
    (GREEN,  "3", "Tools",      "load_mcp_tools(session)",           IA_X + IA_W*0.83),
]
for color, num, title, sub, sx in step_data:
    ax.text(sx - 0.50, STP_Y + STP_H - 0.22, num, fontsize=7.5, color=WHITE,
            ha="center", va="center", fontweight="bold", zorder=6,
            bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.22"))
    txt(sx, STP_Y + STP_H*0.60, title, sz=8.5, w="bold")
    txt(sx, STP_Y + STP_H*0.24, sub, sz=6.5, c=GRAY)

# 2d ReAct Loop
rbox(IA_X, LOP_Y, IA_W, LOP_H, WHITE, PURPLE, lw=1.4)
txt(cx, LOP_Y + LOP_H - 0.30, "ReAct Loop", sz=10, w="bold")
loop_rows = [
    ("LLM picks the next tool to call",    GRAY,  "normal"),
    ("search_web(query, max_results)",      GRAY,  "normal"),
    ("save_note(title, content, topic)",    GRAY,  "normal"),
    ("read_notes(topic)",                   GRAY,  "normal"),
    ("generate_report(topic)",              GRAY,  "normal"),
    ("Final answer  ──────────────",        DARK,  "bold"),
]
for i, (line, color, weight) in enumerate(loop_rows):
    txt(IA_X + 0.22, LOP_Y + LOP_H - 0.72 - i*0.39, line,
        sz=8, c=color, w=weight, ha="left")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — MCP SERVER SECTION
# ══════════════════════════════════════════════════════════════════════════════
rbox(COL_MCP_X, SEC_Y, COL_MCP_W, SEC_H, "#ECFDF5", TEAL, lw=2.0, z=1)
txt(COL_MCP_X + 0.18, SEC_Y + SEC_H - 0.28,
    "FastMCP Server  ·  /mcp", sz=9.5, w="bold", c=TEAL, ha="left")

mc = IM_X + IM_W/2  # centre-x of MCP inner boxes

# 3a Tools
rbox(IM_X, TOOL_Y, IM_W, TOOL_H, WHITE, GREEN)
txt(mc, TOOL_Y + TOOL_H - 0.30, "Tools", sz=10, w="bold", c=GREEN)
tool_rows = [
    ("search_web(query, max_results=5)",  "→ web search via DuckDuckGo"),
    ("save_note(title, content, topic)",  "→ writes data/notes/{topic}/*.md"),
    ("read_notes(topic)",                 "→ reads all notes for the topic"),
    ("generate_report(topic)",            "→ saves data/{topic}_report.md"),
]
for k, (sig, desc) in enumerate(tool_rows):
    ry = TOOL_Y + TOOL_H - 0.72 - k*0.52
    txt(IM_X + 0.15, ry,       sig,  sz=7.8, w="bold", c=DARK, ha="left")
    txt(IM_X + 0.15, ry-0.22,  desc, sz=6.8, c=GRAY,   ha="left")

# 3b Prompts
rbox(IM_X, PRMT_Y, IM_W, PRMT_H, WHITE, AMBER)
txt(mc, PRMT_Y + PRMT_H - 0.30, "Prompts", sz=10, w="bold", c=AMBER)
prmt_rows = [
    ("research_prompt(topic)",     "→ 5-step systematic research workflow"),
    ('summarize_prompt(style)',    '→ "bullet" | "paragraph" | "executive"'),
]
for k, (sig, desc) in enumerate(prmt_rows):
    ry = PRMT_Y + PRMT_H - 0.72 - k*0.55
    txt(IM_X + 0.15, ry,       sig,  sz=7.8, w="bold", c=DARK, ha="left")
    txt(IM_X + 0.15, ry-0.22,  desc, sz=6.8, c=GRAY,   ha="left")

# 3c Resources
rbox(IM_X, RESC_Y, IM_W, RESC_H, WHITE, RED)
txt(mc, RESC_Y + RESC_H - 0.30, "Resources", sz=10, w="bold", c=RED)
resc_rows = [
    ("config://agent",  "→ agent name, personality & style"),
    ("notes://all",     "→ all notes across every topic folder"),
]
for k, (uri, desc) in enumerate(resc_rows):
    ry = RESC_Y + RESC_H - 0.72 - k*0.52
    txt(IM_X + 0.15, ry,       uri,  sz=7.8, w="bold", c=DARK, ha="left")
    txt(IM_X + 0.15, ry-0.22,  desc, sz=6.8, c=GRAY,   ha="left")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — EXTERNAL SERVICES
# ══════════════════════════════════════════════════════════════════════════════
ec = COL_EXT_X + COL_EXT_W/2   # centre-x

# OpenAI API
rbox(COL_EXT_X, OAI_Y, COL_EXT_W, OAI_H, WHITE, BLUE, lw=1.8)
txt(ec, OAI_Y + OAI_H*0.65, "OpenAI API", sz=9.5, w="bold")
txt(ec, OAI_Y + OAI_H*0.28, "api.openai.com · gpt-4.1-mini", sz=7.5, c=GRAY)

# DuckDuckGo
rbox(COL_EXT_X, DDG_Y, COL_EXT_W, DDG_H, WHITE, GREEN, lw=1.8)
txt(ec, DDG_Y + DDG_H*0.65, "DuckDuckGo", sz=9.5, w="bold")
txt(ec, DDG_Y + DDG_H*0.28, "DDGS  ·  no API key needed", sz=7.5, c=GRAY)

# Data Store
rbox(COL_EXT_X, DAT_Y, COL_EXT_W, DAT_H, WHITE, TEAL, lw=1.8)
txt(ec, DAT_Y + DAT_H - 0.30, "Data Store", sz=9.5, w="bold")
store_lines = [
    ("data/notes/{topic}/",          DARK, "bold"),
    ("  core_concepts.md",           GRAY, "normal"),
    ("  applications.md",            GRAY, "normal"),
    ("  key_players.md",             GRAY, "normal"),
    ("",                             GRAY, "normal"),
    ("data/{topic}_report.md",       DARK, "bold"),
    ("",                             GRAY, "normal"),
    ("data/config.json",             DARK, "bold"),
]
for k, (line, col, wt) in enumerate(store_lines):
    txt(COL_EXT_X + 0.12, DAT_Y + DAT_H - 0.70 - k*0.44,
        line, sz=7.2, c=col, w=wt, ha="left")

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS — computed from layout constants so they always align
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. User → Agent  (horizontal at mid-height of User box) ─────────────────
arrow_h(COL_USER_X + COL_USER_W, COL_AGT_X, 4.75, BLUE, "topic")

# ── 2. LLM ↔ OpenAI  (routed above all sections) ───────────────────────────
#    Up from LLM top-centre → across at y=9.0 → down to OpenAI top-centre
LLM_CX = IA_X + IA_W/2                    # 5.60
OAI_CX = COL_EXT_X + COL_EXT_W/2         # 16.35
OAI_TOP = OAI_Y + OAI_H                   # 8.55
CORRIDOR = 9.0                             # above all section boxes (max top 8.90)

# Forward: LLM → OpenAI
arrow_path(
    [(LLM_CX, LLM_Y + LLM_H),  # LLM top centre
     (LLM_CX, CORRIDOR),         # up to corridor
     (OAI_CX, CORRIDOR),         # right
     (OAI_CX, OAI_TOP)],         # down to OpenAI top
    BLUE,
    lbl="API calls + tool schemas",
    lbl_pt=((LLM_CX + OAI_CX)/2, CORRIDOR + 0.10)
)

# Return: OpenAI → LLM (dashed, slightly offset)
LLM_RX = IA_X + IA_W          # right edge of LLM box (= 8.35)
OAI_RX = COL_EXT_X            # left edge of OpenAI (= 15.0)
RETURN_Y = OAI_Y + OAI_H*0.35  # ~7.61

xs = [OAI_RX, RETURN_Y, LLM_RX]   # not right — build path manually:
ax.plot(
    [OAI_CX, OAI_CX, LLM_RX],
    [OAI_Y,  RETURN_Y, RETURN_Y],
    color=BLUE, lw=1.4, linestyle="dashed", zorder=5
)
ax.annotate("", xy=(LLM_RX, RETURN_Y), xytext=(LLM_RX + 0.4, RETURN_Y),
    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=11), zorder=6)
txt((OAI_CX + LLM_RX)/2, RETURN_Y - 0.18, "tool_calls / content",
    sz=6.8, c=BLUE, w="normal")

# ── 3. MCP Client → MCP Server  (horizontal) ────────────────────────────────
MCP_CX  = IA_X + IA_W         # right edge of MCP Client box (8.35)
MCP_SX  = COL_MCP_X           # left edge of MCP Server section (8.90)
MCP_MID = MCP_Y + MCP_H/2     # mid-height of MCP Client (6.15)
arrow_h(MCP_CX, MCP_SX, MCP_MID, TEAL, "HTTP  /mcp")

# ── 4. Agent internal verticals ───────────────────────────────────────────────
arrow_v(cx, LLM_Y,         MCP_Y + MCP_H, PURPLE)   # LLM bottom → MCP top
arrow_v(cx, MCP_Y,         STP_Y + STP_H, PURPLE)   # MCP bottom → Steps top
arrow_v(cx, STP_Y,         LOP_Y + LOP_H, PURPLE)   # Steps bottom → Loop top

# ── 5. Tools → DuckDuckGo  (horizontal at DuckDuckGo mid-height) ─────────────
TOOL_RX = IM_X + IM_W                     # right edge of Tools box (14.35)
DDG_MID = DDG_Y + DDG_H/2                 # mid-height of DDG (6.025)
arrow_h(TOOL_RX, COL_EXT_X, DDG_MID, GREEN, "search")

# ── 6. Tools → Data Store  (L-path: right from Tools, down, right to DataStore)
#    From Tools right-edge at y=TOOL_Y+0.5 (bottom area)
#    → right to x=14.75 (just left of ext col)
#    → down to y=DAT_Y+DAT_H*0.75 (upper area of DataStore)
#    → right to DataStore left edge
TOOL_WRITE_Y = TOOL_Y + 0.60              # 6.20 — within Tools box
GAP_X        = COL_EXT_X - 0.25          # 14.75 — gap corridor
DAT_WRITE_Y  = DAT_Y + DAT_H*0.72        # ~3.68 — within DataStore

arrow_path(
    [(TOOL_RX,     TOOL_WRITE_Y),   # Tools right edge
     (GAP_X,       TOOL_WRITE_Y),   # right into gap corridor
     (GAP_X,       DAT_WRITE_Y),    # down to DataStore level
     (COL_EXT_X,   DAT_WRITE_Y)],   # right into DataStore
    TEAL,
    lbl="write .md", lbl_pt=(GAP_X + 0.35, (TOOL_WRITE_Y + DAT_WRITE_Y)/2)
)

# ── 7. Resources → Data Store  (horizontal read arrow) ───────────────────────
RESC_RX  = IM_X + IM_W                   # right edge of Resources (14.35)
DAT_READ_Y = DAT_Y + DAT_H*0.25          # 1.54 — lower area of DataStore (matches Resources height)
arrow_h(RESC_RX, COL_EXT_X, DAT_READ_Y, RED, "read", lbl_dy=-0.22)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([0.3, 17.7], [0.28, 0.28], color=LGRAY, lw=1.0)
legend_items = [
    (BLUE,   "User / OpenAI"),
    (PURPLE, "Agent Layer"),
    (TEAL,   "MCP Server / Data"),
    (GREEN,  "Tools / DuckDuckGo"),
    (AMBER,  "Prompts"),
    (RED,    "Resources"),
]
for i, (lc, ll) in enumerate(legend_items):
    lx = 0.50 + i * 2.88
    ax.add_patch(FancyBboxPatch(
        (lx, 0.06), 0.24, 0.14,
        boxstyle="round,pad=0.02", facecolor=lc, edgecolor="none", zorder=4))
    txt(lx + 0.36, 0.13, ll, sz=7.5, c=GRAY, ha="left")

plt.tight_layout(pad=0.3)
plt.savefig("assets/architecture.png", dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved -> assets/architecture.png")
