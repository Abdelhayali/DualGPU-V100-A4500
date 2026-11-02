"""
Prints the results tables (markdown) from results/tune-<model>.jsonl and results/hard/*.json.

  python tools/report.py                 every model
  python tools/report.py qwen3.8-27b     one model
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATS = ["coding", "reasoning", "ifeval", "search", "agent", "vision", "longctx"]


def tune_rows(model):
    f = ROOT / "results" / f"tune-{model}.jsonl"
    if not f.exists():
        return []
    rows = {}
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["config"]] = r             # the latest run of each config wins
    return list(rows.values())


def ctx_of(args):
    a = args.split()
    return int(a[a.index("-c") + 1]) if "-c" in a else None


def main():
    models = sys.argv[1:] or sorted(p.stem[5:] for p in (ROOT / "results").glob("tune-*.jsonl"))
    for model in models:
        print(f"### {model}: launch configs\n")
        print("| config | ctx | tok/s (code / chat) | MTP accept | VRAM MiB (A4500+V100) | image | 7k prompt | status |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(tune_rows(model), key=lambda r: -(r.get("mean") or 0)):
            mean = f"**{r['mean']}** ({r['code']} / {r['chat']})" if r.get("mean") else "-"
            acc = f"{r['accept']}%" if r.get("accept") else "-"
            ctx = ctx_of(r["args"])
            lp = (f"ok @ {r['pp']} t/s" if r.get("longprompt") else "FAIL") if "longprompt" in r else "-"
            print(f"| {r['config']} | {ctx // 1024 if ctx else '?'}k | {mean} | {acc} | {r.get('vram_peak_mib') or r.get('vram_mib') or '-'} "
                  f"| {r.get('image', '-')} | {lp} | {r['status']} |")
        hard = sorted((ROOT / "results" / "hard").glob(f"{model}@*.json"))
        if hard:
            print(f"\n### {model}: hard suite\n")
            print("| config | overall | " + " | ".join(CATS) + " | gen tok/s | prompt tok/s |")
            print("|---|---|" + "---|" * len(CATS) + "---|---|")
            for f in hard:
                d = json.loads(f.read_text(encoding="utf-8-sig"))
                s = d["scores"]
                print(f"| {d['label'].split('@', 1)[1]} | **{d['overall']}** | " + " | ".join(str(s.get(c, '-')) for c in CATS)
                      + f" | {d['speed']['gen_tps']} | {d['speed']['prompt_tps']} |")
        print()


if __name__ == "__main__":
    main()
