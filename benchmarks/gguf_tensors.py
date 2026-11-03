#!/usr/bin/env python3
"""List all tensors (name, ggml type, dims) of a GGUF file. Cheap, no model load."""
import struct, sys

NAMES = {0:"F32",1:"F16",2:"Q8_0",3:"Q4_0",4:"Q4_1",5:"Q5_0",6:"Q5_1",7:"Q2_K",8:"Q3_K",
9:"Q4_K",10:"Q5_K",11:"Q6_K",12:"Q2_0",13:"Q1_0",14:"IQ2_XXS",15:"IQ2_XS",16:"IQ2_S",
17:"IQ3_XXS",18:"IQ3_S",19:"IQ4_XS",20:"IQ4_NL",21:"BF16",22:"Q8_K",23:"F8_E4M3",
24:"F64",25:"I8",26:"I16",27:"I32",28:"I64",29:"Q4_NL",30:"Q3_0",31:"TQ1_0",32:"TQ2_0",
33:"MXFP4",34:"NVFP4",35:"F8_E5M2",36:"F8_E4M3",37:"F8_E8M0",38:"F8_FNUZ_E4M3",
39:"Q11_K",40:"Q12_K",41:"IQ1_S",42:"IQ1_M"}

def read_kv(f):
    key_len = struct.unpack("<Q", f.read(8))[0]; key = f.read(key_len).decode("utf-8","replace")
    vtype = struct.unpack("<I", f.read(4))[0]
    if vtype == 8:
        n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
    elif vtype == 9:
        elem = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
        fmt = {1:"B",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}.get(elem)
        if fmt: f.read(struct.calcsize(fmt)*n)
        elif elem == 8:
            for _ in range(n):
                sl = struct.unpack("<Q", f.read(8))[0]; f.read(sl)
        else: raise SystemExit("arr elem %d"%elem)
    elif vtype in (1,2,3,4,5,6,7,10,11,12):
        f.read(struct.calcsize({1:"B",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[vtype]))

f = open(sys.argv[1],"rb")
magic,ver = struct.unpack("<II", f.read(8)); assert magic==0x46554747
n_tensors,n_kv = struct.unpack("<QQ", f.read(16))
for _ in range(n_kv): read_kv(f)
out=[]
for _ in range(n_tensors):
    nlen = struct.unpack("<Q", f.read(8))[0]; name=f.read(nlen).decode()
    ndims = struct.unpack("<I", f.read(4))[0]
    dims = list(reversed(struct.unpack(f"<{ndims}Q", f.read(8*ndims))))
    ty = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    out.append((name, NAMES.get(ty, f"TYPE{ty}"), dims))
out.sort()
for name,ty,dims in out:
    print(f"{name:60} {ty:10} {dims}")
