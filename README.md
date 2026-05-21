# torch_fplib

A PyTorch reimplementation of the **Gaussian Overlap Matrix (GOM)** atomic
fingerprint, originally provided by the C library
[libfp](https://github.com/Rutgers-ZRG/libfp).
torch_fplib runs on CPU and GPU, supports automatic differentiation through
the full fingerprint pipeline, and provides exact derivatives (forces, stress)
without materializing the dense fingerprint Jacobian.

## Why this exists

* **Autograd.** Every step — neighbor cell shifts, Gaussian overlap matrix
  construction, eigendecomposition — is differentiable. Forces and stress fall
  out of `torch.autograd.grad` directly.
* **GPU acceleration.** Batched neighbor and GOM kernels move per-atom
  Python loops into vectorized tensor ops. A single CUDA call processes
  thousands of structures at once via `get_lfp_fast_batch`.
* **Correct analytical stress.** The reference C library's analytical strain
  derivative (`dfpe`) is reliable only on cells with zero off-diagonal stress.
  On sheared cells the off-diagonal Voigt components are off by 57 % – 1625 %
  (see *Validation* below). torch_fplib computes stress via the autograd
  strain parametrization `cell' = (I + ε) · cell`, which is exact for every
  Voigt component and matches single-component finite differences to ≈ 1e-8.
* **VJP-based force/stress projection.** For applications that need
  `Jᵀ · ∂L/∂fp` (CSP search, FP-targeted relaxation, FP-guided MD),
  one backward pass replaces the `nat × fp_dim` passes a dense Jacobian
  would require.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. Optional test extras:

```bash
pip install -e ".[test]"
```

## Quickstart

```python
import torch_fplib
import ase.build

atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)

# Cell format: (lattice, positions, types, znucl)
#   types are 1-indexed atom-type integers; znucl is their atomic numbers.
cell = (
    atoms.cell.array,
    atoms.get_positions(),
    [1] * len(atoms),
    [14],
)

fp = torch_fplib.get_lfp(cell, cutoff=6.0, orbital="s", natx=64)
print(fp.shape)  # (nat, natx)
```

### Force / stress via autograd

```python
import torch
lat = torch.tensor(atoms.cell.array, dtype=torch.float64, requires_grad=True)
pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, requires_grad=True)

fp = torch_fplib.get_lfp((lat, pos, [1]*len(atoms), [14]), cutoff=6.0)

# Example loss: distance to a target fingerprint
L = ((fp - fp_target) ** 2).sum()

dL_dpos, dL_dlat = torch.autograd.grad(L, (pos, lat))
```

### Hungarian-matched FP distance

```python
d = torch_fplib.get_fp_dist(fp_a, fp_b, types=[1]*len(atoms))
```

## API overview

| Function | Use |
|---|---|
| `get_lfp(cell, …)` | Long GOM fingerprint, single structure |
| `get_lfp_fast(cell, …)` | Vectorized single-structure path (GPU-friendly) |
| `get_sfp(cell, …)` | Short (contracted) fingerprint |
| `get_lfp_batch(cells, …)` | Simple loop over multiple structures |
| `get_lfp_fast_batch(cells, …)` | One batched GPU call across all atoms of all structures |
| `get_lfp_from_ase_neighbors(...)` | Reuse an ASE neighbor list — avoids redundant search |
| `get_fp_dist(fp1, fp2, types)` | Hungarian-matched per-type fingerprint distance |

The `cell` argument everywhere is `(lat, rxyz, types, znucl)`:

* `lat` — (3, 3) lattice vectors
* `rxyz` — (nat, 3) Cartesian positions, Å
* `types` — (nat,) 1-indexed atom-type integers
* `znucl` — (ntyp,) atomic numbers for each type

## Validation

### Stress (autograd) vs. C `dfpe`

Tested on a deliberately sheared CdSe cell (all six Voigt stress components
non-zero):

| Quantity | Method | Max abs. error |
|---|---|---|
| Force (per atom) | C libfp `dfp` (analytic) | 1.14 × 10⁻¹⁰ |
| Stress (Voigt, diagonal) | C libfp `dfpe` (analytic) | ≈ 1 % |
| Stress (Voigt, off-diagonal) | C libfp `dfpe` (analytic) | **57 % – 1625 %** |
| Stress (all 9 strain components) | torch_fplib autograd | **7.69 × 10⁻⁹** |

The autograd path is validated against single-component finite differences,
not just the diagonal — every component of `∂E/∂ε` matches FD to ≈ 1e-8.

> **Tip.** The strain gradient `∂E/∂ε` is *not* symmetric. The correct Voigt
> mapping is `σ_v = ∂E/∂ε [a_v, b_v] / V` with
> `_VOIGT_IDX = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]`. Symmetrizing or
> doubling off-diagonal strains will give a factor-of-four bug.

### Performance

Batched eigvalsh on small GOM blocks is the dominant cost. On A100-40GB,
double precision, 2000 atoms:

| Operation | CPU | GPU | Speedup |
|---|---|---|---|
| Fingerprint (forward) | 1.0× | 1.7× | GPU wins |
| Fingerprint Jacobian (backward) | 1.0× | 1.5× | GPU wins |

At ≤ 1024 atoms the per-call overhead lets CPU catch up; for `dFP` at 1024
atoms CPU is actually about 10 % faster because the eigvalsh backward kernel
is CPU-friendly at that size. torch_fplib's `get_lfp_fast_batch` includes an
adaptive CPU fallback (batch ≤ 256, n ≤ 64) calibrated against this crossover.

In end-to-end benchmarks (training equivariant MLIPs that consume GOM
features inside the model), the PyTorch implementation has consistently run
faster than the C reference, primarily because batched GPU eigvalsh is much
faster than per-atom CPU calls.

## Used in

* **CRISP** — fingerprint-space crystal structure prediction
  (uses VJP for FP-targeted relaxation, CAWR, and FP-Jacobian mutations).
* **PALLAS** — phase-transition pathway prediction with dimer + FP distance
  metric.
* **EosNet v2** — differentiable GOM features inside an e3nn equivariant
  MLIP backbone.

If you use torch_fplib in published work, please cite the original libfp
fingerprint construction (Sadeghi, Goedecker *et al.*) and link back to this
repository.

## License

MIT — see [`LICENSE`](LICENSE).
