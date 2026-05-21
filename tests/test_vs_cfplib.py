"""Validate torch-fplib against C libfp on real structures."""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_fplib

# Auto-detect CdSe data path
_CANDIDATES = [
    "/Users/li/dev/FpGNN/CdSe_E",
    "/scratch/lz432/FpGNN/CdSe_E",
]
STRUCT_DIR = None
for p in _CANDIDATES:
    if os.path.isdir(p):
        STRUCT_DIR = p
        break


def load_structure(vasp_file):
    from ase.io import read as ase_read
    from functools import reduce
    atoms = ase_read(vasp_file)
    lat = atoms.cell[:]
    rxyz = atoms.get_positions()
    chem_nums = list(atoms.numbers)
    znucl = reduce(lambda re, x: re + [x] if x not in re else re, chem_nums, [])
    types = [znucl.index(z) + 1 for z in chem_nums]
    return (np.array(lat), np.array(rxyz), np.array(types), np.array(znucl))


def test_correctness():
    """Compare torch-fplib (original + fast) vs C libfp on a single structure."""
    import libfp

    vasp_file = os.path.join(STRUCT_DIR, "Cd16Se16_1.vasp")
    cell = load_structure(vasp_file)
    cutoff = 6.0
    natx = 300

    fp_c = np.array(libfp.get_lfp(cell, cutoff=cutoff, natx=natx, log=False, orbital='s'))

    # torch original (CPU f64)
    fp_orig = torch_fplib.get_lfp(cell, cutoff=cutoff, natx=natx, orbital='s',
                                   device='cpu', dtype=torch.float64).numpy()

    # torch fast (CPU f64)
    fp_fast = torch_fplib.get_lfp_fast(cell, cutoff=cutoff, natx=natx,
                                        device='cpu', dtype=torch.float64).numpy()

    print(f"Shape: C {fp_c.shape}, orig {fp_orig.shape}, fast {fp_fast.shape}")
    print(f"Atom 0 top-5 C:    {fp_c[0,:5]}")
    print(f"Atom 0 top-5 orig: {fp_orig[0,:5]}")
    print(f"Atom 0 top-5 fast: {fp_fast[0,:5]}")

    diff_orig = np.max(np.abs(fp_c - fp_orig))
    diff_fast = np.max(np.abs(fp_c - fp_fast))
    print(f"Max diff orig vs C: {diff_orig:.2e}")
    print(f"Max diff fast vs C: {diff_fast:.2e}")

    # CUDA if available
    if torch.cuda.is_available():
        fp_cuda = torch_fplib.get_lfp_fast(cell, cutoff=cutoff, natx=natx,
                                            device='cuda', dtype=torch.float64).cpu().numpy()
        diff_cuda = np.max(np.abs(fp_c - fp_cuda))
        print(f"Max diff CUDA vs C: {diff_cuda:.2e}")

    ok = diff_orig < 1e-6 and diff_fast < 1e-6
    print("PASS" if ok else "FAIL")


def test_benchmark():
    """Benchmark: per-structure loop vs batched across structures."""
    import libfp

    N_list = [50, 200, 500, 2598]

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load all structures once
    all_cells = []
    for i in range(1, max(N_list) + 1):
        f = os.path.join(STRUCT_DIR, f"Cd16Se16_{i}.vasp")
        if os.path.exists(f):
            all_cells.append(load_structure(f))

    header = f"{'N':>5} | {'C':>12} | {'fast-CPU':>12}"
    if has_cuda:
        header += f" | {'batch-CUDA64':>12} | {'batch-CUDA32':>12} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for N in N_list:
        cells = all_cells[:N]
        if not cells:
            continue
        n = len(cells)

        # C libfp
        t0 = time.time()
        for c in cells:
            libfp.get_lfp(c, cutoff=6.0, natx=300, log=False, orbital='s')
        t_c = time.time() - t0

        # torch fast CPU (per-structure loop)
        t0 = time.time()
        for c in cells:
            torch_fplib.get_lfp_fast(c, cutoff=6.0, natx=300, device='cpu', dtype=torch.float64)
        t_fast_cpu = time.time() - t0

        line = f"{n:5d} | {t_c:7.2f}s {t_c/n*1000:4.1f}ms | {t_fast_cpu:7.2f}s {t_fast_cpu/n*1000:4.1f}ms"

        if has_cuda:
            # Batched CUDA f64
            torch.cuda.synchronize()
            t0 = time.time()
            fps64 = torch_fplib.get_lfp_fast_batch(
                cells, cutoff=6.0, natx=300, device='cuda', dtype=torch.float64)
            torch.cuda.synchronize()
            t_batch64 = time.time() - t0

            # Batched CUDA f32
            torch.cuda.synchronize()
            t0 = time.time()
            fps32 = torch_fplib.get_lfp_fast_batch(
                cells, cutoff=6.0, natx=300, device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            t_batch32 = time.time() - t0

            best = min(t_batch64, t_batch32)
            line += f" | {t_batch64:7.2f}s {t_batch64/n*1000:4.1f}ms | {t_batch32:7.2f}s {t_batch32/n*1000:4.1f}ms | {t_c/best:5.2f}x"

        print(line)

    # Verify batch correctness
    if has_cuda:
        print("\nBatch correctness check:")
        cells_50 = all_cells[:50]
        fps_batch = torch_fplib.get_lfp_fast_batch(
            cells_50, cutoff=6.0, natx=300, device='cuda', dtype=torch.float64)

        # Compare batch vs single-structure (both CUDA)
        max_diff_vs_single = 0.0
        worst_i_single = 0
        for i, c in enumerate(cells_50):
            fp_single = torch_fplib.get_lfp_fast(
                c, cutoff=6.0, natx=300, device='cuda', dtype=torch.float64)
            fp_b = fps_batch[i]
            d = torch.max(torch.abs(fp_single - fp_b)).item()
            if d > max_diff_vs_single:
                max_diff_vs_single = d
                worst_i_single = i

        # Compare batch vs C
        max_diff_vs_c = 0.0
        worst_i_c = 0
        for i, c in enumerate(cells_50):
            fp_c = np.array(libfp.get_lfp(c, cutoff=6.0, natx=300, log=False, orbital='s'))
            fp_b = fps_batch[i].cpu().numpy()
            d = np.max(np.abs(fp_c - fp_b))
            if d > max_diff_vs_c:
                max_diff_vs_c = d
                worst_i_c = i

        print(f"  Batch vs single-CUDA (50 structs): {max_diff_vs_single:.2e} (worst: struct {worst_i_single})")
        print(f"  Batch vs C          (50 structs): {max_diff_vs_c:.2e} (worst: struct {worst_i_c})")

        # Debug worst structure
        if max_diff_vs_single > 1e-10:
            i = worst_i_single
            fp_s = torch_fplib.get_lfp_fast(cells_50[i], cutoff=6.0, natx=300, device='cuda', dtype=torch.float64)
            fp_b = fps_batch[i]
            print(f"  Struct {i} single top-5: {fp_s[0,:5].cpu().numpy()}")
            print(f"  Struct {i} batch  top-5: {fp_b[0,:5].cpu().numpy()}")
            print(f"  Struct {i} shapes: single {fp_s.shape}, batch {fp_b.shape}")

        ok = max_diff_vs_c < 1e-6
        print("  PASS" if ok else "  FAIL")


if __name__ == "__main__":
    if STRUCT_DIR is None:
        print("ERROR: CdSe_E data not found")
        sys.exit(1)
    print(f"Data: {STRUCT_DIR}\n")

    print("=" * 60)
    print("Test 1: Correctness")
    print("=" * 60)
    test_correctness()

    print("\n" + "=" * 60)
    print("Test 2: Speed benchmark")
    print("=" * 60)
    test_benchmark()
