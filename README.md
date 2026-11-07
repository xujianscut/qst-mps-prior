# Learned measurement-conditioned priors for matrix-product-state tomography

Code, results and hardware data for the paper *Learned measurement-conditioned priors for
matrix-product-state tomography* (J. Xu, D. Zeng, J. Paisley, Q. Zhao).

A network trained on simulated states maps a fixed local Pauli protocol to MPS cores. Used on its
own it does not beat a tuned maximum-likelihood (MLE) fit; used as a gauge-invariant prior inside
the likelihood fit (MAP refinement), gated by a chi-squared goodness-of-fit test of its own
estimate, it improves on tuned MLE at every system size and shot budget tested.

## Layout

| path | contents |
|---|---|
| `experiments/` | the experiments of the paper's main results (MAP refinement, gate, baselines, hardware, cost) |
| `legacy/` | the original pipeline: ablations of the amortized network alone and the unconditional generative prior over cores (see `legacy/README.md`) |
| `results/` | every result JSON the paper's tables and figures are generated from |
| `data/hw_local.npz` | IBM `ibm_aachen` measurement records: 5 states, 63 local Paulis, 4096 shots, with circuit angles and ideal values |
| `paper/` | `gen_tables.py` and `fig_map.py`, which regenerate the paper's tables and main figure from `results/` |

## Requirements

Python 3.10, PyTorch 2.5 with CUDA, NumPy, Matplotlib. The hardware acquisition script
(`legacy/hw_local.py`) additionally needs `qiskit` and `qiskit-ibm-runtime` and reads an IBM
Quantum token from the environment variable `IBM_TOKEN`; it is never written to disk.

Scripts write JSON results and network checkpoints to `$QST_OUT` (default `results/`); the hardware
evaluation reads `$QST_HW` (default `data/hw_local.npz`). Run from `experiments/` or put it on
`PYTHONPATH`.

## Experiments and where they appear

| script | paper |
|---|---|
| `amortized.py` | network alone: point fidelity, prior-only and shuffled controls, ensemble intervals (table 1, table 4 ensemble column) |
| `mle_baselines.py` | MLE sweep over restarts and iterations with residual and true-fidelity restart selection, bootstrap and fixed-scale intervals, direct readout from bitstrings, classical shadows (sections on the amortized network and on observables, table 4) |
| `map_refine.py` | network, tuned MLE, warm start, MAP refinement and family-mean prior on identical records, hyperparameters chosen on a validation set; paired differences; interval coverage under shift (table 2, figure 1, tables 1 and 3) |
| `gate.py` | chi-squared gate: in distribution, under family shift and under shot shift (tables 2 and 3) |
| `hardware_map.py` | network, MLE and MAP on the hardware records |
| `cost.py` | per-state compute of the network, MLE and MAP |
| `ising_family_scan.py` | diversity, entanglement and gap of candidate disordered Ising families |
| `summarize.py` | prints a summary of all `rescue_*.json` files |

Typical invocations (one GPU each):

```bash
cd experiments
python amortized.py     --n 6 --design local --meas indep          # or --meas basis
python mle_baselines.py --n 6 --design local --meas indep
python map_refine.py    --n 10 --shots 50 --latency 0              # --family tfim for the Ising family
python gate.py          --n 10 --shots 200                         # needs the map_refine result first
python hardware_map.py
python ../paper/gen_tables.py ../results ../paper && python ../paper/fig_map.py ../results ../paper/fig_map.pdf
```

Result files are named `rescue_n{n}_{design}_S{shots}[_basis][_seed{k}][_tfim].json` (MAP study),
`gate_chi2_n{n}_S{shots}_q0.9[_tfim].json`, `cond_*.json` (network alone) and `mle_*.json` (MLE study).

## Measurement models

* `indep`: each design observable carries independent Gaussian noise of variance (1 - <P>^2)/S.
* `basis`: the nine qubit-wise settings (qubit q in basis a if q is even, c if odd, (a, c) in {X, Y, Z}^2)
  are simulated, S bitstrings are sampled per setting, and every design observable is estimated from
  the bitstrings of all settings that cover it, as on hardware.

The copy budget of the default `local` design is 9S for every n.

## License

MIT, see `LICENSE`.
