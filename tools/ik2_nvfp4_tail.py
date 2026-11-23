"""Inkling-2-Mini (unsloth's inkling2 layout, the one Maya reads): re-quantize the Q8_0 expert tensors to NVFP4 with
per-expert .scale tensors, the format unsloth's UD-Q4_K_XL already uses for layers 0-19.

UD-Q4_K_XL keeps the experts of layers 20-23 in Q8_0 (4.0 GiB); in NVFP4 they take 2.1 GiB, so on a 20 GB card Maya
keeps every expert in VRAM at a longer context (with room for its drafter). Every other tensor is copied byte for byte.

NVFP4 as ggml stores it: blocks of 64 values, a UE4M3 scale per 16 (ggml_fp32_to_ue4m3(amax / 6)), 4-bit E2M1 codes
(kvalues_mxfp4, the nearest one), and a per-expert F32 scale; unsloth's convention, kept here: the expert's largest
magnitude maps to 6 * 448 = 2688, scale = amax / 2688.

  python tools/ik2_nvfp4_tail.py <in.gguf> <out.gguf> [--layers 20,21,22,23]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kernel" / "src-inklings" / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import NVFP4, dequantize  # noqa: E402

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.float32)   # E2M1 magnitudes doubled (kvalues_mxfp4, codes 0..7)


def quant_nvfp4(x: np.ndarray) -> np.ndarray:
    """x [rows, K] float32 (K % 64 == 0), already divided by the expert scale -> NVFP4 bytes [rows, K / 64 * 36]."""
    rows, k = x.shape
    sub = x.reshape(-1, 16)
    amax = np.abs(sub).max(axis=1)
    ue = NVFP4.fp32_to_ue4m3(amax / 6.0)                      # one byte per 16 values
    d = NVFP4.ue4m3_to_fp32(ue).astype(np.float32)            # with the 0.5 of the doubled kvalues
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d[:, None] > 0, np.abs(sub) / d[:, None], 0.0)
    idx = np.abs(r[..., None] - KV).argmin(axis=-1).astype(np.uint8)           # nearest magnitude (first on ties)
    code = idx | np.where((sub < 0) & (idx > 0), np.uint8(8), np.uint8(0))     # sign bit (ggml: -0 is code 0)
    code = code.reshape(-1, 4, 16)
    qs = (code[:, :, :8] | (code[:, :, 8:] << 4)).reshape(-1, 32)              # x[j] | x[8 + j] << 4 per 16
    blocks = np.concatenate([ue.reshape(-1, 4), qs], axis=1)                   # 4 scale bytes, then 32 code bytes
    return blocks.reshape(rows, k // 64 * 36)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", default="20,21,22,23")
    a = ap.parse_args()
    layers = {int(v) for v in a.layers.split(",")}
    r = GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():                          # every key as it is (gguf_new_metadata.py's way)
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(name, f.contents(), vt, sub_type=sub)
    todo, err = [], []
    for t in r.tensors:
        lay = int(t.name.split(".")[1]) if t.name.startswith("blk.") else -1
        if lay in layers and t.name.endswith("_exps.weight") and t.tensor_type == Q.Q8_0:
            todo.append(t)
            ne = [int(v) for v in t.shape]                   # [K, rows, experts]
            w.add_tensor_info(t.name, (ne[2], ne[1], ne[0] // 64 * 36), np.dtype(np.uint8),
                              ne[2] * ne[1] * (ne[0] // 64 * 36), raw_dtype=Q.NVFP4)
            w.add_tensor_info(t.name.replace(".weight", ".scale"), (ne[2],), np.dtype(np.float32), ne[2] * 4)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
    print(f"{arch}: {len(todo)} Q8_0 expert tensors -> NVFP4 (layers {sorted(layers)})", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        if t in todo:
            ne = [int(v) for v in t.shape]
            out = np.empty((ne[2], ne[1], ne[0] // 64 * 36), dtype=np.uint8)
            scale = np.empty(ne[2], dtype=np.float32)
            for e in range(ne[2]):
                x = dequantize(np.asarray(t.data[e]), Q.Q8_0).astype(np.float32)   # [rows, K]
                amax = float(np.abs(x).max())
                s = amax / 2688.0 if amax > 0 else 1.0
                out[e] = quant_nvfp4(x / s)
                scale[e] = s
                if e % 32 == 0:                               # the error against Q8_0, a sample
                    back = dequantize(out[e], Q.NVFP4) * s
                    err.append(float(np.sqrt(np.mean((back - x) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)))
            w.write_tensor_data(out)
            w.write_tensor_data(scale)
            print(f"  {t.name}: NVFP4, relative RMS error {np.mean(err[-4:]):.4f}", flush=True)
        else:
            w.write_tensor_data(np.asarray(t.data))
    w.close()
    print(f"wrote {a.dst}: mean relative RMS error of the re-quantized experts {np.mean(err):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
