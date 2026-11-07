# Original pipeline (amortized estimator alone, unconditional prior)

Code for the results on the amortized network alone and on the unconditional generative prior. Every such number comes from one of the
scripts below. All scripts are single-file PyTorch programs with an `argparse` interface and
write their results as JSON next to themselves; the plotting scripts read those JSONs and
emit the figure PDFs.

## Requirements

python 3.10, torch 2.5 (CUDA), numpy, matplotlib. The hardware script additionally needs
`qiskit` and `qiskit-ibm-runtime`, and reads an IBM Quantum token from the environment
variable `IBM_TOKEN` (it is never written to disk).

Each script takes `--dev cuda:0`; isolate a GPU with `CUDA_VISIBLE_DEVICES=<g>`.

## Approach B — the amortized estimator (main method)

| script | what it produces |
|---|---|
| `qst_cond.py` | the fixed-protocol conditional estimator: headline fidelities, the k=0 prior-only and shuffled controls, the design ladder (`local1`/`local`/`local3`/`random`), the encoder ablation (set-encoder vs plain MLP), the diversity and shot sweeps, the n = 6/8/10 scaling, the bond-dimension and depth sweeps, and (with `--strengthen`) the k-NN and dense-state-MLP baselines |
| `qst_calibration.py` | MC-dropout ensemble + **per-observable** split-conformal calibration (one multiplier Q_O per observable, at the exact finite-sample rank), reliability curves, and the calibrated predictions for the 18 observables the protocol never measures |
| `mle_design.py` | classical prior-free per-state MPS maximum likelihood on the same local values, across the design ladder |
| `angle_baseline.py`, `angle_baseline_fid.py` | known-ansatz baselines: regress the 2n circuit angles from the same 63 observables, under angle-MSE and under our gauge-invariant state-fidelity loss |
| `qst_native.py`, `qst_bign.py` | native O(n·chi^3) transfer-matrix contraction; the n = 16, 20 memory measurements (never forms the 2^n state vector) |
| `hw_local.py` | IBM hardware run: prepares 5 held-out states, measures the 63 local Paulis in one batched `EstimatorV2` job, saves `hw_local.npz`. Feed it back with `qst_cond.py --hw_eval` for the closed-loop reconstruction |

Typical invocations:

```bash
# headline + controls (shallow-circuit family, informative local design)
python qst_cond.py --dev cuda:0 --n 6 --family shallow --design local --steps 30000

# design ladder / encoder ablation / strengthened baselines
python qst_cond.py --dev cuda:0 --family shallow --design local1     # or local3, random
python qst_cond.py --dev cuda:0 --family shallow --design local --arch mlp
python qst_cond.py --dev cuda:0 --family shallow --design local --strengthen

# calibration + unmeasured-observable intervals (both families)
python qst_calibration.py --dev cuda:0 --family random  --design local
python qst_calibration.py --dev cuda:0 --family shallow --design local

# classical MPS-MLE head-to-head, known-ansatz baselines
python mle_design.py
python angle_baseline.py && python angle_baseline_fid.py

# hardware closed loop (needs IBM_TOKEN in the environment)
IBM_TOKEN=... python hw_local.py
python qst_cond.py --dev cuda:0 --family shallow --design local --hw_eval hw_local.npz
```

## Approach A — the generative-prior route (complementary, appendix)

| script | what it produces |
|---|---|
| `qst_mps5.py` | flow-matching prior over canonicalized MPS cores + measurement-guided (proximal / Doob h-transform) posterior sampling; the classical baselines it is compared against (MPS-MLE, generic-prior pCN, classical shadows) |
| `posterior_val.py` | gold-standard check: self-normalized importance sampler from the learned prior reweighted by the exact shot likelihood, versus the guided sampler |
| `qst_review.py` | out-of-distribution stress test and the MLE-convergence sanity checks |
| `qst_prioronly.py` | the k=0 prior-only and shuffled-measurement integrity controls, on concentrated vs diverse families |
| `qst_gauge.py` | gauge-stability experiment behind the canonicalization discussion |

## Figures

`plot_cond.py` (main, design, control, agnostic, scale, calibration), `plot_scaling.py`
(n-scaling), `plot_extra.py`, `plot_postval.py`, `plot_review.py` (OOD), `plot_tfim.py`,
`regen_clip.py`. They read the JSONs in `results/` — move that directory's contents next to
the scripts (or edit the `D` path at the top of each plotting script) before running.

## Notes on the measurement model

* `k` is the number of Pauli **observables** given to the estimator (63 for the default local
  design at n = 6); those group into `m = 9` qubit-wise-commuting **settings**, so the copy
  budget is `m·S`, not `k·S`.
* The synthetic experiments draw each observable's error independently as
  `N(0, (1 - <P>^2)/S)`. This matches the per-observable variance of running the m settings at
  S shots each, but idealizes away the correlations among Paulis that share a setting's shots.
  The hardware run does not use that idealization: all 63 observables go into one PUB per
  state and co-measured groups are estimated from shared shots.
