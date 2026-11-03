"""Rebuild the repository's vector diagrams without external dependencies."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)


def text(x, y, value, size=18, color="#334155", weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(value)}</text>'


def box(x, y, width, height, title, lines, accent="#0f766e"):
    value = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="white" stroke="#cbd5e1"/>'
    value += f'<rect x="{x}" y="{y + 20}" width="4" height="{height - 40}" rx="2" fill="{accent}"/>'
    value += text(x + 20, y + 34, title, 19, "#0f172a", 650)
    for i, line in enumerate(lines):
        value += text(x + 20, y + 63 + i * 24, line, 15)
    return value


def arrow(path, color="#64748b", dashed=False):
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)"' + (' stroke-dasharray="7 5"' if dashed else '') + '/>'


def svg(width, height, content):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>
<g font-family="Segoe UI, Arial, sans-serif">{content}</g></svg>'''


hero = '<rect width="1400" height="390" rx="24" fill="#101d2d"/>'
hero += '<circle cx="1320" cy="0" r="340" fill="#123a42" opacity=".65"/>'
hero += text(65, 62, "INFORMATION RETRIEVAL  /  UIT DSC 2026", 16, "#73d5c1", 650)
hero += text(65, 132, "Vietnamese Legal", 52, "#f8fafc", 700)
hero += text(65, 203, "Retrieval", 68, "#f8fafc", 700)
hero += text(68, 247, "Complementary retrieval. Focused evidence. Audited ranking.", 22, "#cbd5e1")
for x, label in [(68, "BM25"), (188, "Dense retrieval"), (397, "LambdaMART"), (594, "Top5")]:
    width = {68: 105, 188: 194, 397: 182, 594: 100}[x]
    hero += f'<rect x="{x}" y="284" width="{width}" height="42" rx="21" fill="#20384b"/>'
    hero += text(x + 18, 311, label, 17, "#e2e8f0", 550)
hero += '<path d="M900 90 V300" stroke="#47606d"/>'
hero += text(963, 133, "0.93383", 64, "#f0c878", 700)
hero += text(967, 174, "Reported public Recall@5", 21, "#e2e8f0", 600)
hero += text(967, 209, "Historical 05A / 06A branch", 17, "#b6c8d6")
hero += text(967, 262, "Owner-reported competition result", 16, "#b6c8d6")
hero += text(967, 289, "Demo uses fictional examples", 16, "#b6c8d6")
hero += text(68, 365, "ncat218  ·  Multi-stage legal document search", 14, "#9db2c4")
(OUT / "hero.svg").write_text(svg(1400, 390, hero), encoding="utf-8")

diagram = '<rect width="1400" height="750" rx="20" fill="#f4f7f8"/>'
diagram += text(40, 55, "Two ranking paths, one final Top5", 32, "#0f172a", 700)
diagram += text(40, 88, "Historical 05A / 06A architecture: shortlist quality and inherited final scores are separate dependencies.", 17)
diagram += text(40, 151, "CANDIDATE & SHORTLIST PATH", 14, "#0f766e", 700)
diagram += box(40, 180, 230, 142, "Three retrieval lanes", ["BM25 · BGE-M3 · DEk21", "Natural Top200 per lane", "No gold insertion"])
diagram += box(310, 180, 210, 142, "RRF → Top50", ["Equal source weights", "1 / (60 + source rank)", "Document-ID tie break"])
diagram += box(560, 180, 240, 142, "Jina v3.5", ["Joint-list Top50 scoring", "Title + selected chunks", "Pinned model / evidence"])
diagram += box(840, 180, 240, 142, "04B LambdaMART", ["Retrieval + Jina features", "Frozen shortlist ranker", "Select Top15"])
diagram += box(1120, 180, 235, 142, "Top15 document IDs", ["Final candidates", "Check inherited coverage", "Backfill required in 06A"])
for start, end in [(270, 310), (520, 560), (800, 840), (1080, 1120)]:
    diagram += arrow(f"M{start} 250 H{end - 5}")
diagram += text(40, 409, "INHERITED SCORE & DECISION PATH", 14, "#8a661c", 700)
diagram += box(40, 440, 300, 142, "17C / 17D context", ["Protected multi-source pool", "Qwen evidence and scores", "14A baseline anchors"], "#c79535")
diagram += box(395, 440, 300, 142, "Frozen 17E ensemble", ["Retrieval + Qwen features", "Full scoring context retained", "Required pairs completed"], "#c79535")
diagram += box(760, 440, 310, 142, "Final selection", ["Keep exact baseline Top2", "Fill 3 slots from Top15", "Use 17E scores + fixed ties"], "#c79535")
diagram += box(1120, 440, 235, 142, "Top5", ["Unique document IDs", "Score / fallback audit", "Validated submission"], "#c79535")
diagram += arrow("M340 510 H390") + arrow("M695 510 H755") + arrow("M1070 510 H1115")
diagram += arrow("M1237 322 V370 H915 V435")
diagram += text(955, 359, "shortlist IDs", 14)
diagram += arrow("M170 582 V624 H880 V587", dashed=True)
diagram += text(355, 650, "Exact protected prefix from the inherited baseline", 15)
diagram += text(40, 705, "05A: any missing shortlist score → whole-query baseline fallback.    06A: backfill first; incomplete scores stop inference.", 16)
(OUT / "architecture.svg").write_text(svg(1400, 750, diagram), encoding="utf-8")
print("Built assets/hero.svg and assets/architecture.svg")
