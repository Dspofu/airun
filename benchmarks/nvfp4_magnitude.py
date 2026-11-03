#!/usr/bin/env python3
"""Decisive NVFP4 magnitude test.

Faithfully replicate airun's dequant (doubled E2M1 codebook x halved UE4M3
scale, block = [4 scales][32 code bytes], low nibble first) on a REAL weight
tensor, then compare the value distribution:
   raw        = per-block dequant (what airun computes today)
   x .scale   = raw * the tensor's *.scale
   x 1/.scale = raw / the tensor's *.scale
A real weight matrix has magnitudes ~O(0.01..10), mean ~0, no overflow.
Whichever interpretation is sane tells us how (or whether) the .scale is used.
"""
import struct, sys, statistics

path = sys.argv[1]
f = open(path, "rb")
magic, ver = struct.unpack("<II", f.read(8)); assert magic == 0x46554747
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

def read_string():
    ln = struct.unpack("<Q", f.read(8))[0]; return f.read(ln).decode("utf-8","replace")
def read_value():
    tag = struct.unpack("<I", f.read(4))[0]
    if tag == 8:
        ln = struct.unpack("<Q", f.read(8))[0]; f.read(ln)
    elif tag == 9:
        elem = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
        if elem in (1,2,3,4,5,6,7,10,11,12):
            f.read({1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[elem]*n)
        elif elem == 8:
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]; f.read(ln)
        else: raise SystemExit(f"elem {elem}")
    elif tag in (1,2,3,4,5,6,7,10,11,12):
        f.read({1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[tag])
    else: raise SystemExit(f"tag {tag}")
for _ in range(n_kv): read_string(); read_value()

rows=[]
for _ in range(n_tensors):
    name = read_string()
    ndims = struct.unpack("<I", f.read(4))[0]
    dims = list(struct.unpack(f"<{ndims}Q", f.read(8*ndims)))
    ty = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    rows.append((name,dims,ty,off))
data_start = (f.tell()+31) & ~31

def find(pred):
    for r in rows:
        if pred(r): return r
    return None

# a real NVFP4 weight and its .scale
w = find(lambda r: r[2]==40 and r[0].endswith(".weight") and "ffn_down" in r[0])
name,dims,ty,off = w
sname = name[:-len(".weight")]+".scale"
s = find(lambda r: r[0]==sname)
f.seek(data_start+s[3]); gscale = struct.unpack("<f", f.read(4))[0]
print(f"weight {name} dims={dims}\n.scale {sname} = {gscale:.6g}\n")

E2M1 = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]  # airun doubled codebook
def ue4m3(x):
    if x==0 or x==0x7F: return 0.0
    e=(x>>3)&0xF; m=x&0x7
    if e==0: return m/1024.0
    return (1+m/8)*2.0**(e-7)*0.5
def deq_block(blk):
    d=blk[:4]; qs=blk[4:36]; out=[]
    for sub in range(4):
        dv=ue4m3(d[sub]); q=qs[sub*8:sub*8+8]
        for i in range(8):
            out.append(dv*E2M1[q[i]&0xF]); out.append(dv*E2M1[q[i]>>4])
    return out

elems = 1
for d in dims: elems*=d
nblocks = elems//64
import random; random.seed(2)
sample = sorted(random.sample(range(nblocks), min(400, nblocks)))
raw=[]
for b in sample:
    f.seek(data_start+off + b*36)
    raw.extend(deq_block(f.read(36)))
raw = raw[:40000]
xs = [v*gscale for v in raw]
inv = [v/gscale for v in raw]
def rep(label, vals):
    av=[abs(v) for v in vals]
    print(f"{label:14} min={min(vals):12.4g} max={max(vals):12.4g} mean={statistics.fmean(vals):10.4g} "
          f"std={statistics.pstdev(vals):10.4g} maxAbs={max(av):10.4g}")
rep("raw", raw)
rep("raw*gscale", xs)
rep("raw/gscale", inv)
print("\nA sane weight matrix: maxAbs ~0.01..10, mean ~0, no inf/nan.")
