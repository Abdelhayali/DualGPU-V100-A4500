"""
Looks up size + sha256 (Hugging Face LFS metadata) for every file in the "files" list of configs/<model>.json and keeps
them in configs/files.json, so the GUI can download and verify missing files. Run once after adding a model:
  python tools/fill-hashes.py <model id> [...]
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "configs" / "files.json"
known = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
api = HfApi()

for mid in sys.argv[1:]:
    cfg = json.loads((ROOT / "configs" / f"{mid}.json").read_text(encoding="utf-8"))
    for f in cfg["files"]:
        rev = f.get("revision") or next((x.get("revision") for x in cfg["files"] if x["repo"] == f["repo"] and x.get("revision")), None)
        info = api.get_paths_info(f["repo"], [f["file"]], revision=rev, expand=True)[0]
        known[f"{mid}/{f['file']}"] = {"repo": f["repo"], "size": info.size, "sha256": info.lfs.sha256,
                                       "url": hf_hub_url(f["repo"], f["file"], revision=rev)}
        local = ROOT / "models" / mid / f["file"]
        ok = "on disk, size ok" if local.exists() and local.stat().st_size == info.size else "NOT on disk / size differs"
        print(f"{mid}: {f['file']} {info.size} {info.lfs.sha256[:12]}  {ok}")

out.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n", encoding="utf-8")
