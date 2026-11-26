#!/usr/bin/env python3
"""Graft the MTP (nextn) layer blk.<N>.* of a donor Qwen3.8-27B GGUF into a target GGUF that has none
(NariQ). Target tensors are copied byte for byte; block_count becomes N+1 and nextn_predict_layers 1.
Per-layer metadata arrays of length N are extended with their last value. No dependencies.
Usage: mtp_graft.py <target.gguf> <donor.gguf> <out.gguf> [--head] [--list]
  --head  also copies the donor output.weight / token_embd.weight as blk.N.nextn.shared_head_head / embed_tokens (needed for NariQ, whose LM head is banded)
"""
import os, struct, sys, json

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def parse(path):
    size = os.path.getsize(path)
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
    rs = lambda: f.read(struct.unpack("<Q", f.read(8))[0])
    kvs = []  # (key, type, value, raw_bytes)
    for _ in range(nkv):
        p0 = f.tell()
        k = rs().decode()
        t = struct.unpack("<I", f.read(4))[0]
        if t == 8:
            v = rs()
        elif t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            if et == 8:
                v = ("arr", et, n, [rs() for _ in range(n)])
            else:
                v = ("arr", et, n, list(struct.unpack(f"<{n}{FMT[et]}", f.read(SZ[et] * n))))
        else:
            v = struct.unpack("<" + FMT[t], f.read(SZ[t]))[0]
        p1 = f.tell(); f.seek(p0); raw = f.read(p1 - p0)
        kvs.append((k, t, v, raw))
    keys = {k: v for k, t, v, _ in kvs}
    align = int(keys.get("general.alignment", 32))
    ts = []
    for _ in range(nt):
        n = rs(); nd = struct.unpack("<I", f.read(4))[0]
        d = struct.unpack(f"<{nd}Q", f.read(8 * nd)); ty, off = struct.unpack("<IQ", f.read(12))
        ts.append([n, d, ty, off])
    base = (f.tell() + align - 1) // align * align
    order = sorted(range(nt), key=lambda i: ts[i][3])
    for a, b in zip(order, order[1:] + [None]):
        ts[a].append((ts[b][3] if b is not None else size - base) - ts[a][3])
    return dict(f=f, ver=ver, kvs=kvs, keys=keys, align=align, ts=ts, base=base)


def kv_bytes(k, t, v):
    kb = k.encode(); out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t)
    if t == 8:
        return out + struct.pack("<Q", len(v)) + v
    if t == 9:
        _, et, n, items = v
        out += struct.pack("<IQ", et, len(items))
        if et == 8:
            return out + b"".join(struct.pack("<Q", len(x)) + x for x in items)
        return out + struct.pack(f"<{len(items)}{FMT[et]}", *items)
    return out + struct.pack("<" + FMT[t], v)


def main():
    tgt, don, out = sys.argv[1:4]
    T = parse(tgt); D = parse(don)
    arch = T["keys"]["general.architecture"].decode()
    nb = int(T["keys"][f"{arch}.block_count"])
    if "--list" in sys.argv:
        for k, t, v, _ in T["kvs"]:
            if not k.startswith("tokenizer."):
                print(k, t, (v[:3] if isinstance(v, tuple) else v) if not isinstance(v, tuple) else ("arr", v[1], v[2], v[3][:8]))
        print("tensors", len(T["ts"]), "blk.%d:" % nb, [t[0] for t in T["ts"] if t[0].startswith(b"blk.%d." % nb)])
        print("types", sorted({t[2] for t in T["ts"]}))
        return
    assert not any(t[0].startswith(b"blk.%d." % nb) for t in T["ts"]), "target already has an MTP layer"
    dnb = int(D["keys"][f"{arch}.block_count"]) - 1
    pre = b"blk.%d." % dnb
    new = [t for t in D["ts"] if t[0].startswith(pre)]
    if "--head" in sys.argv:  # also copy the donor LM head as the MTP layer's own head (target head is banded / missing)
        new += [[pre + b"nextn.shared_head_head.weight"] + t[1:] for t in D["ts"] if t[0] == b"output.weight"]
        new += [[pre + b"nextn.embed_tokens.weight"] + t[1:] for t in D["ts"] if t[0] == b"token_embd.weight"]
    assert new, "donor has no MTP layer"
    kvs = []
    seen_nextn = False
    for k, t, v, raw in T["kvs"]:
        if k == f"{arch}.block_count":
            kvs.append(kv_bytes(k, t, nb + 1)); continue
        if k == f"{arch}.nextn_predict_layers":
            kvs.append(kv_bytes(k, t, 1)); seen_nextn = True; continue
        if t == 9 and v[2] == nb and v[1] != 8 and not k.startswith("tokenizer."):
            kvs.append(kv_bytes(k, t, ("arr", v[1], nb + 1, v[3] + [v[3][-1]]))); print("extended", k); continue
        kvs.append(raw)
    if not seen_nextn:
        kvs.append(kv_bytes(f"{arch}.nextn_predict_layers", 4, 1))
    al = T["align"]; pad = lambda n: (-n) % al
    rows = []; o = 0
    for n, d, ty, off, sz in T["ts"]:
        rows.append((n, d, ty, o)); o += sz + pad(sz)
    for n, d, ty, off, sz in new:
        rows.append((b"blk.%d." % nb + n[len(pre):], d, ty, o)); o += sz + pad(sz)
    tmp = out + ".partial"
    with open(tmp, "wb") as w:
        w.write(b"GGUF" + struct.pack("<IQQ", T["ver"], len(rows), len(kvs)))
        for b in kvs: w.write(b)
        for n, d, ty, a in rows:
            w.write(struct.pack("<Q", len(n)) + n + struct.pack("<I", len(d)) + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", ty, a))
        w.write(b"\0" * pad(w.tell()))
        for src, lst in ((T, T["ts"]), (D, new)):
            f = src["f"]
            for n, d, ty, off, sz in lst:
                f.seek(src["base"] + off); left = sz
                while left:
                    c = f.read(min(left, 16 << 20)); w.write(c); left -= len(c)
                w.write(b"\0" * pad(sz))
    os.replace(tmp, out)
    print(json.dumps({"out": out, "gb": round(os.path.getsize(out) / 1e9, 3), "added": [r[0].decode() for r in rows[len(T["ts"]):]]}, indent=1))


main()
