# Regenerates src/iq_grids.rs and ../airun-cuda/kernels/iq_grids.cuh: the IQ2_S,
# IQ3_XXS and IQ3_S lattice codebooks, copied out of ggml's ggml-common.h.
#
#   python gen_iq_grids.py path/to/ggml/src/ggml-common.h
import os, re, sys

src = open(sys.argv[1], encoding='utf-8').read()
here = os.path.dirname(os.path.abspath(__file__))

def table(name):
    m = re.search(r'GGML_TABLE_BEGIN\((\w+),\s*' + name + r',\s*(\w+)\)(.*?)GGML_TABLE_END\(\)', src, re.S)
    ty, size, body = m.group(1), m.group(2), m.group(3)
    vals = re.findall(r'0x[0-9a-fA-F]+', body)
    assert len(vals) == int(size), (name, len(vals), size)
    return ty, int(size), vals

names = ['iq2s_grid', 'iq3xxs_grid', 'iq3s_grid']
rs_ty = {'uint64_t': 'u64', 'uint32_t': 'u32'}
head = ("// Generated from ggml's ggml-common.h (MIT, the ggml authors) by\n"
        "// gen_iq_grids.py: the lattice codebooks of the IQ2_S, IQ3_XXS and IQ3_S\n"
        "// types. Each entry packs 8 (IQ2) or 4 (IQ3) unsigned byte magnitudes;\n"
        "// the signs travel separately in the block. Do not edit by hand.\n")

rs = [head]
cu = [head, '#pragma once\n#include <stdint.h>\n']
for n in names:
    ty, size, vals = table(n)
    rows = ''.join('    ' + ', '.join(vals[i:i + 8]) + ',\n' for i in range(0, size, 8))
    rs.append(f'pub(crate) static {n.upper()}: [{rs_ty[ty]}; {size}] = [\n{rows}];\n\n')
    cu.append(f'static __device__ const {ty} {n}[{size}] = {{\n{rows}}};\n\n')
open(os.path.join(here, 'src', 'iq_grids.rs'), 'w', encoding='utf-8', newline='\n').write(''.join(rs))
open(os.path.join(here, '..', 'airun-cuda', 'kernels', 'iq_grids.cuh'), 'w', encoding='utf-8', newline='\n').write(''.join(cu))
