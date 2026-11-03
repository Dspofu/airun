#!/usr/bin/env python3
"""Dump the architecture metadata keys of a GGUF file (v3: u32 mtype tags, u8 elem tags)."""
import struct, sys

f = open(sys.argv[1], "rb")
magic, ver = struct.unpack("<II", f.read(8)); assert magic == 0x46554747
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
print(f"version={ver}")

def rstr():
    ln = struct.unpack("<Q", f.read(8))[0]; return f.read(ln).decode("utf-8", "replace")

def rval():
    tag = struct.unpack("<I", f.read(4))[0]
    if tag == 8:
        ln = struct.unpack("<Q", f.read(8))[0]; return f.read(ln).decode("utf-8", "replace")
    if tag == 9:
        e = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if e in (1,2,3,4,5,6,7,10,11,12):
            fmt = {1:"B",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[e]
            return list(struct.unpack(f"<{n}{fmt}", f.read(struct.calcsize(fmt)*n)))
        if e == 8:
            out=[]
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]; out.append(f.read(ln).decode("utf-8","replace"))
            return out
        raise SystemExit(f"elem {e}")
    if tag in (1,2,3,4,5,6,7,10,11,12):
        fmt = {1:"B",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[tag]
        return struct.unpack("<"+fmt, f.read(struct.calcsize(fmt)))[0]
    raise SystemExit(f"tag {tag}")

for _ in range(n_kv):
    k = rstr(); v = rval()
    kl = k.lower()
    if any(s in kl for s in ("nextn", "moe", "block_count", "arch", "ssm", "linear_attn",
                             "full_attention", "expert", "layer", "head", "n_embd", "ffn",
                             "conv", "vocab", "rope", "norm", "scale", "quant", "mtp")):
        print(f"{k:55} = {v}")
