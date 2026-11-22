"""Reasoning-on vs reasoning-off on the hard suite: writes docs/reasoning-on.png and prints a Markdown table.

Each pair is (name, reasoning-off result, reasoning-on result) under results/hard. A missing reasoning-on file is skipped.
Usage: python tools/reasoning_chart.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
HARD = ROOT / "results" / "hard"

PAIRS = [
    ("Surogate3.7-35B-A3B Q5_K_M", "surogate-3.7-35b@l60-mtp3-128k", "surogate-3.7-35b@l60-mtp3-128k-thinking-on"),
    ("Qwen3.6-35B-A3B UD-Q5_K_XL", "qwen3.6-35b-a3b@auto-mtp3-128k", "qwen3.6-35b-a3b@auto-mtp3-128k-thinking-on"),
    ("Qwen3.8-27B UD-Q5_K_XL", "qwen3.8-27b@l60-mtp3-128k", "qwen3.8-27b@l60-mtp3-128k-thinking-on"),
    ("Inkling-2-Mini IQ4_NL", "inkling-2-mini@a4500-32k-q8-min", "inkling-2-mini@a4500-32k-q8-high-thinking-on"),
    ("Qwen3.8-40B Ultimate Heretic", None, "qwen3.8-40b-heretic@auto-mtp2-64k"),
    ("Apex-35B-V1.0 Q4_K_M (text only)", "apex-35b-v1@l60-nodraft-262k", "apex-35b-v1@l60-nodraft-262k-thinking-on"),
]
# Heretic's reasoning-off run (80.7) was not kept as a result file: it reasons anyway, without closing </think>.
OFF_OVERRIDE = {"Qwen3.8-40B Ultimate Heretic": 80.7}
CATS = [("coding", "coding"), ("reasoning", "reasoning"), ("ifeval", "instr"), ("search", "search"),
        ("agent", "agent"), ("vision", "vision"), ("longctx", "longctx")]


def load(stem):
    p = HARD / f"{stem}.json"
    return json.loads(p.read_text(encoding="utf-8")) if stem and p.exists() else None


def main():
    rows = []
    for name, off_stem, on_stem in PAIRS:
        on = load(on_stem)
        if not on:
            continue
        off = load(off_stem)
        off_overall = off["overall"] if off else OFF_OVERRIDE.get(name)
        rows.append((name, off_overall, on))
    rows.sort(key=lambda r: -r[2]["overall"])

    print("| # | model | gen tok/s | " + " | ".join(c for _, c in CATS) + " | **overall on** | off | change |")
    print("|---" * (len(CATS) + 6) + "|")
    for i, (name, off_overall, on) in enumerate(rows, 1):
        s = on["scores"]
        cells = " | ".join(f"{s[k]:.0f}" for k, _ in CATS)
        delta = f"{on['overall'] - off_overall:+.1f}" if off_overall is not None else "-"
        off_txt = f"{off_overall:.1f}" if off_overall is not None else "-"
        print(f"| {i} | {name} | {on['speed']['gen_tps']:.1f} | {cells} | **{on['overall']:.1f}** | {off_txt} | {delta} |")

    names = [r[0] for r in rows][::-1]
    on_v = [r[2]["overall"] for r in rows][::-1]
    off_v = [r[1] or 0 for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(rows) + 1.4), dpi=130)
    y = range(len(rows))
    h = 0.38
    ax.barh([i + h / 2 for i in y], on_v, h, label="reasoning on (5x token budget)", color="#2a6fdb")
    ax.barh([i - h / 2 for i in y], off_v, h, label="reasoning off", color="#b8c4d6")
    for i, (a, b) in enumerate(zip(on_v, off_v)):
        ax.text(a + 0.6, i + h / 2, f"{a:.1f}", va="center", fontsize=9, color="#1b3f7a", fontweight="bold")
        if b:
            ax.text(b + 0.6, i - h / 2, f"{b:.1f}", va="center", fontsize=8, color="#555")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("hard suite overall (72 tasks)")
    ax.set_title("Hard suite: reasoning on vs off (RTX A4500 + Tesla V100)", fontsize=11, pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    out = ROOT / "docs" / "reasoning-on.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
