# Issues found in upstream qm3 (sergio-marti/ren-qm3), to report

These came up while validating QMCube for BS-DFT/MM of a [4Fe-4S] enzyme (APS reductase, 35,817 atoms) on Leonardo, October 2026. The fork `Carlos-A-Ramos-G/ren-qm3`, branch `gpu-string`, fixes the PySCF/GPU path. The other engines still have the link-atom problem (section 1).

Status: **not reported yet**. Upstream base: commit `8dd0010`.

---

## 1. `Link_grad` is not the derivative of the energy (all engines except `pyscf.run_pbc`)

**Where:** `qm3/engines/__init__.py`, `Link_grad( lnk, grd )`. It is called by `orca`, `gaussian`, `nwchem`, `psi4`, `qchem`, `tmole`, `quick`, `xtb`, `dftb`, `dftd4`, `mopac`, `sqm` and `pyscf.run` (CPU). On the fork, only `pyscf.run_pbc` uses the fix.

**What is wrong.** The link H is placed by `Link_coor` at `H = Q + d (M − Q)/|M − Q|` (d = 1.1 Å, Q = QM host, M = MM partner), so its position depends on both Q and M. `Link_grad` only removes part of the bond-parallel component of the link-H gradient from the host:

```python
t = sum( vec[k] * ( grd[3*qm_i+k] - grd[3*mm_j+k] ) ) * 0.5 / m
grd[3*qm_i:3*qm_i+3] = grd[3*qm_i:3*qm_i+3] - t * vec / m
```

The gradient on the link H itself is then dropped. For example, `orca.parse_log` copies only the first `len(sel)` atoms into `mol.grad`, so the MM partner M never gets its share. Check each engine's copy loop when porting. The result is not dE/dR. The forces on Q are wrong, and M feels no force from the QM region through the link.

**Evidence** (qm3 CHARMM sample, central finite differences of the QM/MM energy):

| | Error on host Q | Force on partner M |
|---|---|---|
| `Link_grad` | up to **13 kJ/(mol·Å)** | **missing** |
| `Link_chain` (fix) | ≤ 2×10⁻⁴ kJ/(mol·Å) | correct |

**Fix (already in the fork, commit `f15477e`).** `Link_chain( mol, lnk, grd, dst )` applies the exact chain rule of `Link_coor`. With u = (M − Q)/r and J = (d/r)(I − u uᵀ):

```
g_Q += g_H − J g_H
g_M += J g_H
```

Porting each engine means passing it the link-H gradients (the rows after `len(sel)` in its QM gradient) and calling `Link_chain( mol, self.lnk, g_link )` instead of `Link_grad( self.vla, g )`.

**Impact.** Any MD, optimization, string or umbrella sampling with link atoms and these engines has non-conservative forces at the QM/MM boundary. The error is local, but it heats the boundary in NVE and biases geometries there.

---

## 2. `orca` engine: points to fix or check before BS-DFT/MM MD with ORCA

Besides `Link_grad` (section 1):

1. **A stale gradient is reused silently.** `get_grad` runs `os.system( self.exe )` without checking the return code, then reads `orca.engrad`. If ORCA fails (SCF not converged, crash, time limit), the gradient from the **previous step** is read as if it were new. Fix:
   - delete `orca.engrad` and `orca.pcgrad` before each run
   - check the exit status, and check for "ORCA TERMINATED NORMALLY" in `orca.out`
2. **Point-charge forces.** ORCA computes `orca.pcgrad` from the total density, so the UKS α-only bug found in ASH (RagnarB83/ash#526) does not apply. This has not been checked by finite differences in qm3 yet; do that on MM atoms near the QM region before MD.
3. **Minimum image (`img = True`).** Point charges are re-imaged around the QM centroid at every call. An MM atom that crosses the half-box boundary jumps, so the energy is discontinuous. In practice, keep the QM region far from the boundary or use a cutoff sphere (`sel_MM`) that never reaches half the box.
4. **Charge precision.** `orca.pc` writes the charges with 4 decimals (`%12.4lf`). After redistributing the link partner's charge over its group, the total MM charge can drift by about 1e-4 e per group. That is harmless, but `%.6lf` costs nothing.
5. **Broken symmetry (feature, not a bug).** ORCA can build the BS state itself:
   ```
   %scf FlipSpin <0-based QM indices of the flipped metals>  FinalMs 0.0 end
   * xyz <charge> <HS multiplicity>
   ```
   With `FlipSpin` in the template, every call re-flips the restart orbitals (`orca.gbw`, read automatically), which would be wrong after the first step. The engine needs a way to switch templates after the first call: first call with `FlipSpin` and the HS multiplicity, later calls at Ms = 0 with no flip. A simple option is an optional `inp_next` argument, or documenting `eng.inp = ...` after the first call. For the string method, each node must run in its own directory so that each keeps its own `orca.gbw`.

---

## 3. `pyscf.runGPU` (fixed in the fork as `run_pbc`)

Shown with `samples/benchmark/compare_upstream.py`: upstream engine vs the fork, on the CHARMM sample and the 35,817-atom [4Fe-4S] system (B3LYP/def2-SVP, A100).

| # | Problem in upstream `runGPU` | Effect | Fix in `run_pbc` |
|---|---|---|---|
| 3.1 | MM coordinates converted to Bohr (`crd *= self.cx`) and then passed with `unit="Angstrom"` | Embedding at the wrong geometry: 30 kJ/mol error on the water test, 632 kJ/mol on CHARMM | Pass Å; wrap MM images around the QM centroid |
| 3.2 | MM forces from a CPU loop over bare point charges, inconsistent with the Ewald energy | Finite-difference error 0.21 (< rcut) / 0.91 (> rcut) kJ/(mol·Å); about 200 s per gradient | `grad_nuc_mm() + grad_hcore_mm(dm) + de_ewald_mm`; FD error about 2×10⁻⁴; 12–14× faster |
| 3.3 | No DFT grid response | 0.09 kJ/(mol·Å) error on QM atoms | `grid_response=True` (default) |
| 3.4 | RKS forced whenever spin = 0 | Broken symmetry at Ms = 0 impossible (`assert dms.ndim == 2`) | `unrestricted` option |
| 3.5 | No reuse of the previous density | SCF from scratch at every step | `dm0` reuse (13–16 cycles per MD step) |
| 3.6 | gpu4pyscf 1.8.1 `dump_chk` gets a cupy `e_tot` (h5py `TypeError`) | Crash on the first step | Not needed in `run_pbc`; a runtime patch exists for the upstream engine |
| 3.7 | Link atoms through `Link_grad` | See section 1 | `Link_chain` |

Benchmark files: `samples/benchmark/compare_upstream.py`, `samples/benchmark/pyscf_upstream.py` (upstream engine, verbatim), `samples/test_pyscf_pbc.py`.

---

## How to report (suggested)

- One issue for section 1, plus a PR that switches all engines to `Link_chain`. The engines differ in how they store link-H gradients, so test each with a finite-difference check on the CHARMM sample, like `samples/test_pyscf_pbc.py`. That is a small, self-contained fix that affects every user.
- One PR with `run_pbc` and the benchmark (section 3). It is larger, so describe the motivation (correct periodic embedding, BS at Ms = 0, GPU speed) in the issue first.
- The ORCA robustness points (2.1–2.4) can go in the section 1 PR or in a small issue. The BS template switch (2.5) is a feature request.
