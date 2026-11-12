# Hard Resets Compose: Exact Time-Parallel Spiking Neurons via Piecewise-Affine Map Scans

Code, raw results and paper sources for the paper of the same name (Dekun Yang, Zhejiang University).

**Idea.** For a hard-reset LIF neuron, the map from incoming membrane potential to outgoing potential over a
segment of `L` steps is affine below a threshold and piecewise constant above it, with at most `L+1` pieces.
These maps are closed under composition, so the hard-reset recurrence is an associative scan in *map space*.
After pruning to the reachable states (whose lower bound is itself a cheap scan), chunk maps have 2–5 pieces
on average. The resulting forward pass is exact, non-iterative, and its depth does not depend on the number of spikes.

## Layout

| Path | Content |
|---|---|
| `lifscan/exact.py` | Exact rational-arithmetic reference: serial recurrence, segment maps and their composition, the closed form (Theorem 1), the tree scan, and a generic piecewise-affine class used to count soft-reset pieces |
| `lifscan/fast.py` | PyTorch prototype (CPU/MPS): two-level scan, pruning, bit-exact repair, affine-scan backward |
| `lifscan/triton_lif.py` | CUDA kernels (Triton): fused serial baselines (plain and time-tiled), `scan_forward_v3` (two-level: envelope bound + pruning + capacity-K maps + overflow fallback + optional repair), `scan_forward_v4` (three-level composition), tiled backward scan, and the autograd layer `lif_tuned` |
| `lifscan/triton_soft.py` | Exact two-level scan for soft-reset (subtractive) LIF neurons, tiled serial baseline, FFT adjoint |
| `lifscan/triton_affine.py` | Exact scan for affine reset `u <- gamma*v + delta` (`0 <= gamma <= 1`; contains hard and soft reset), serial baseline, autograd layer `lif_affine` (gradients w.r.t. inputs, `u0`, `beta`, `gamma`, `delta`) |
| `lifscan/triton_alif.py` | Exact scan for adaptive-threshold (ALIF) neurons on a predicted adaptation box (verified carry, serial fallback, replay/repair), serial baselines, backward pass as two affine scans, autograd layer `alif` |
| `lifscan/shdnet.py` | SHD/SSC data loading and the 700-256-256-20 feed-forward LIF network |
| `tests/` | `test_exact.py`, `test_fast.py` (CPU); `test_triton.py`, `test_triton_tiled.py`, `test_affine.py`, `test_alif.py`, `test_alif_grad.py` (GPU) |
| `experiments/` | All experiment scripts (E1–E26, see below) and the GPU job queues |
| `results/` | Raw JSON results (and server logs in `results/logs/`) behind every number in the paper |
| `paper/` | LaTeX source; `make_numbers.py` regenerates every quoted number from `results/`, `figs/make_figs.py` every figure |

Superseded intermediate results are kept for transparency: `e4_*` (Apple M4 prototype timings), `e6_*` (first GPU
benchmark with an untuned serial baseline) and `e6b_*` (fair benchmark before envelope pruning). The paper uses `e6c_*`.

## Correction in v2 (serial baselines)

The first arXiv version tuned the serial kernels over four tilings only, and its serial backward kernel did not prefetch
inputs. `e26*` show that full grids and a prefetching backward make the serial baselines up to 2.5x (forward) and 2.4x
(backward) faster, with bit-identical outputs. All speed results were re-run on one RTX 5090 with full grids on both
sides (`e6d`, `e9d`, `e11`, `e7c_timing_node5`, `e14`, `e15t`, `e17`-`e21`, `e25d/e`); the headline speedups drop from
7.8x (forward) / 31x (forward+backward) to 5.7x / 16x.  The superseded results (`e6c_*`, `e9_*`, `e7c_timing_node2`,
`e25_alif_*`) are kept for transparency.  The library's serial defaults and dispatch thresholds now follow the new data.

## Reproducing

```bash
pip install -r requirements.txt

# CPU: exact-arithmetic tests and piece counts
python3 tests/test_exact.py
python3 experiments/e1_pieces.py      # hard vs soft pieces (exact), ~10 min
python3 experiments/e1c_tight.py      # pruned piece counts

# GPU (CUDA + Triton)
python3 tests/test_triton.py && python3 experiments/test_v3.py
python3 experiments/e6b_gpu_fair.py 0.9     # best-vs-best benchmark -> results/e6c_fair_beta0.9.json
python3 experiments/e6b_gpu_fair.py 0.99

# SHD: download https://zenkelab.org/datasets/shd_train.h5.gz and shd_test.h5.gz, gunzip
export SHD_DATA=/path/to/shd
python3 experiments/e7_shd.py scan 1024 0 30 64      # mode in {serial, scan, sj}, T, seed, epochs, batch
python3 experiments/e7b_gradcheck.py 1024
python3 experiments/e7c_timing.py
DATASET=ssc python3 experiments/e7_shd.py scan 1024 0 20 64   # SSC (also needs ssc_*.h5)
python3 experiments/e9_threelevel.py 0.9                      # long sequences, two- vs three-level
python3 experiments/e11_auto.py                               # automatic dispatch
python3 experiments/e12_alif_exact.py                         # ALIF post-reset maps (exact, CPU)
python3 experiments/e26_serial_bwd_tune.py      # tuning of the serial baselines (backward) -- see the note below
python3 experiments/e26b_serial_fwd_tune.py     # ... and forward
python3 experiments/e6d_gpu_full.py 0.9         # best-vs-best benchmark with full grids on both sides (supersedes e6b/e6c)
python3 experiments/e9d_long_full.py 0.9        # long sequences, full grids (supersedes e9)
python3 experiments/e24_alif_killtest.py        # ALIF: exact chunk-map sizes vs entry box (CPU)
python3 experiments/e25d_alif_fwd.py results/e25d_alif_fwd.json                                # ALIF scan, forward
python3 experiments/e25e_alif_fb.py results/e25d_alif_fwd.json results/e25e_alif_fb.json      # ALIF, backward + totals
python3 experiments/allen_fetch_ids.py results/allen_cells_78.json && python3 experiments/e15t_allen_timing.py
# Official Allen GLIF models on the same cells (needs allensdk, which requires Python <= 3.11, e.g. via `uv venv --python 3.11`)
# first query the API for the models (see results/allen_glif_models_raw.json), then:
python3 experiments/glif_fetch.py                                  # neuron_config.json of every GLIF level
/path/to/py311/bin/python experiments/e27_glif_official.py 16      # official models simulated with the Allen SDK
python3 experiments/e28_allen_alif.py                              # ALIF fits with the ALIF scan in the loop
python3 experiments/e28t_alif_timing.py && python3 experiments/e27b_metrics.py   # timing; F1 and explained-variance comparison
# Database-wide version (all 1,044 mouse cells with GLIF models; 20 kHz storage; 8 concurrent fits under CUDA MPS):
python3 experiments/glif_query_all.py && ALLEN_OUT=data/allen_all python3 experiments/allen_fetch_all.py 8
python3 experiments/glif_fetch.py results/allen_glif_models_raw_all.json results/allen_glif_models_all.json
bash experiments/queue_database.sh      # E15 (hard/soft LIF) -> E28 (ALIF) -> E27b, after E27 on the database (see the script)
bash experiments/e29_throughput.sh      # throughput of 64 ALIF fits, serial kernels vs scans, under MPS
python3 experiments/e30_typical_pieces.py && python3 experiments/e30b_persistence.py   # Theorem 8: typical map size (CPU)
python3 experiments/e23_reset_lower_bounds.py  # Theorem 7(c): exponential lower bounds for order-reversing / expanding resets (exact, CPU)
python3 experiments/e20_fpt_psn.py        # head-to-head: FPT (re-implemented from the official code) and PSN vs exact soft scan
python3 experiments/e21_chunk_restart.py  # head-to-head: spike-by-spike chunk restarts vs map scan, vs firing rate
python3 experiments/e22_allen_affine.py 0 1   # learned affine reset on Allen cells (E22_WARM=1: warm start; needs data/allen)

# Paper numbers, figures and PDF
python3 paper/make_numbers.py && python3 paper/figs/make_figs.py
cd paper && latexmk -pdf main.tex
```

`experiments/queue.sh` runs the GPU jobs one at a time, so that timings are not contaminated by other jobs.

## Using the layer

```python
import torch
from lifscan.triton_lif import lif_tuned
I = torch.randn(4096, 1024, device="cuda")          # [T, N] input currents (N = batch * neurons)
u0 = torch.zeros(1024, device="cuda")
spikes = lif_tuned(I, u0, beta=0.9, theta=1.0, u_r=0.0, mode="scan")   # or mode="serial"

# recommended: pick serial or scan per pass from (N, T); never slower than serial in our tests
from lifscan.triton_lif import lif_auto
spikes = lif_auto(I, u0, beta=0.9)
```

### Fitting neuron models

```python
from lifscan.fit import lif_param            # gradients w.r.t. I, u0, beta, theta, u_r
spikes = lif_param(I, u0, beta, theta, u_r, mode="auto")
```

`experiments/e14_fit.py` (synthetic parameter recovery) and `experiments/e15_allen_fit.py` (Allen Cell Types
recordings, fetched with `experiments/allen_fetch.py`) reproduce Sec. "Application" of the paper.

The dispatch thresholds (`AUTO_FWD_MAX_N`, `AUTO_FWD_MIN_T`, `AUTO_BWD_MAX_N`, `AUTO_BWD_MIN_T` in
`lifscan/triton_lif.py`) were measured on an RTX 5090; recalibrate them with `experiments/e11_auto.py` on other GPUs.

The scan is faster than a tuned fused serial kernel only when the serial kernel cannot fill the GPU (few
independent recurrences `N`, long `T`). See Fig. 3 of the paper for the measured crossover.

## License

The source code and associated code documentation are released under the [MIT License](LICENSE).
The manuscript text and figure assets in `paper/` are excluded from this code license.

## Citation

```bibtex
@misc{yang2026hardresets,
  title  = {Hard Resets Compose: Exact Time-Parallel Spiking Neurons via Piecewise-Affine Map Scans},
  author = {Yang, Dekun},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://github.com/cheshireyang/hard-resets-compose}
}
```
