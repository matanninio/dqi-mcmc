# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

# DQI-MCMC: Sampling Algorithms for Decoded Quantum Interferometry Experiments

> [!NOTE]
> This repository is associated to a [research publication](https://arxiv.org/abs/2607.28120) and
> the code here is not actively maintained.
> This is not an officially supported IBM Quantum software.

This repository contains the MCMC sampling code used to produce the numerical results in:

> Gil-Fuster, E., Ninio, M., Bittel, L., Shimoni, Y., Eisert, J., Woerner, S., and Carrera Vázquez, A.
> **Approximate sampling from decoded quantum interferometry via Markov chain Monte Carlo methods.**
> *arXiv* (2026). [arXiv:2607.28120](https://arxiv.org/abs/2607.28120)

## 1. Overview

Decoded Quantum Interferometry (DQI) is a quantum algorithm for combinatorial optimisation that
encodes problem structure into a quantum interference pattern. The paper shows that its performance
can be analysed by sampling from a distribution proportional to the square of an optimal polynomial
P evaluated at the objective function f(x), and establishes both capacity results (XOR-SAT) and
hardness results (OPI) for this sampling problem via Markov chain Monte Carlo methods.

This repository implements two Block Gibbs samplers for this task:

- **XOR-SAT** (`BlockGibbsSamplerOptimized`): samples from the MAX-XOR-SAT distribution used
  in the paper's capacity analysis.
- **OPI** (`BlockGibbsSampler`): samples from the Optimal Polynomial Intersection (OPI)
  distribution used in the paper's hardness experiments.

Both samplers run the resampling algorithms (Algorithm 1: Continuous, Algorithm 2: Restart)
described in the paper and record τ\_DQI — the number of Gibbs steps to the first "good sample."

---

## 2. Installation

### Prerequisites

- **Python ≥ 3.10** (3.13 recommended)
- **[uv](https://github.com/astral-sh/uv)** (fast Python package manager) — install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

A C compiler (clang on macOS, gcc on Linux) is needed to build the Cython extension. Both are
available by default on standard development machines.

### Create a virtual environment and install

```bash
# Create a new venv using Python 3.13 (or whichever Python ≥ 3.10 you have)
uv venv --python 3.13

# Install the package in editable mode (builds the Cython extension automatically)
uv pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest` and `ruff`. Omit it if you only want to run experiments.

> **Note on numba / Python 3.13:** if `uv pip install -e .` fails because `llvmlite` cannot
> build, first install a compatible numba version explicitly, then retry:
>
> ```bash
> uv pip install "numba>=0.61"
> uv pip install -e ".[dev]"
> ```

### Tested versions

The package was developed and tested with the following dependency versions on Python 3.13.0:

| Package | Tested version |
|---------|---------------|
| numpy   | 2.4.6         |
| scipy   | 1.18.0        |
| cython  | 3.2.8         |
| gmpy2   | 2.3.1         |
| h5py    | 3.16.0        |
| numba   | 0.66.0        |
| ldpc    | 2.4.1         |

Newer patch versions are expected to work. If you encounter issues, install these exact versions
with `pip install numpy==2.4.6 scipy==1.18.0 gmpy2==2.3.1 h5py==3.16.0 numba==0.66.0 ldpc==2.4.1`.

### Verify the installation

```bash
.venv/bin/python -c "
from dqi_mcmc.api._elementary_poly_fast import build_elementary_poly_cache_fast
print('Cython extension OK')
"
```

### Invoking the commands

After `pip install -e .` (or `uv pip install -e .`) the following console scripts are placed
on your `PATH` inside the virtual environment:

```bash
generate-opi-rhs --help         # OPI problem generation
generate-xorsat-rhs --help      # XOR-SAT problem generation
precompute-xorsat-lookup --help # pre-compute lookup tables
xorsat-sampling --help          # XOR-SAT Block Gibbs sampling
xorsat-resampling --help        # XOR-SAT resampling (collect τ_DQI values)
opi-resampling --help           # OPI resampling (collect τ_DQI values)
sample-opi --help               # single OPI trial — human-readable summary
multisample-opi --help          # collect N good OPI samples, tab-separated output
verify-opi-trajectories --help  # single-batch trajectory inspection
```

Alternatively, the same scripts are available as thin shell wrappers in `bin/` that invoke
the virtual environment's Python directly, so they work without activating the venv:

```bash
bin/generate-opi-rhs --help
bin/generate-xorsat-rhs --help
bin/precompute-xorsat-lookup --help
bin/xorsat-sampling --help
bin/xorsat-resampling --help
bin/opi-resampling --help
bin/sample-opi --help
bin/multisample-opi --help
bin/verify-opi-trajectories --help
```

Every flag accepts both a short (`-x`) and long (`--long-name`) form. The sections below
always show the short form first.

---

## 3. Generating Problem Files

### OPI problems

Generate 100 random RHS matrices for each prime p used in the paper.
All files are saved to `problems/opi/`.

```bash
# Primes used in the paper
for p in 13 17 19 23 29 31 37 41 43 47; do
    bin/generate-opi-rhs -p $p
done
```

Each file is named `rhs_nsamples100_p{p}_r{r}.npy` where `r = p // 2`.

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-p` | `--p` | Prime field size | required |
| `-r` | `--r` | Entries per row (r) | `p // 2` |
| `-n` | `--n-samples` | RHS vectors to generate | 100 |
| `-s` | `--seed` | Random seed | 123 |
| `-o` | `--output` | Output `.npy` path | `problems/opi/rhs_nsamples{n}_p{p}_r{r}.npy` |

### XOR-SAT problems

Generate 100 random RHS vectors for each problem size m used in the paper.
All files are saved to `problems/maxxorsat/`.

```bash
for m in 100 250 500 750 1000 3000; do
    bin/generate-xorsat-rhs -m $m
done
```

Each file is named `rhs_nsamples100_m{m}_p2_r1.npy`.

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-m` | `--m` | Number of constraints | required |
| `-n` | `--n-samples` | RHS vectors to generate | 100 |
| `-s` | `--seed` | Random seed | 123 |
| `-o` | `--output` | Output `.npy` path | `problems/maxxorsat/rhs_nsamples{n}_m{m}_p2_r1.npy` |

---

## 4. Pre-computing XOR-SAT Lookup Tables

The XOR-SAT sampler uses a pre-computed lookup table of log P²(f(x)) values.
This one-time step is required before running XOR-SAT sampling.

```bash
# Pre-compute for each m (fast for small m; m=3000 takes a few minutes)
for m in 100 250 500 750 1000 3000; do
    bin/precompute-xorsat-lookup -m $m
done
```

Tables are saved to `dqi_mcmc/data/lookup_tables/maxxorsat/` and loaded automatically
by the sampler. Pass `--force` to recompute an existing table.

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-m` | `--m` | Problem size | required |
| `-l` | `--ell` | Polynomial degree | `(n + 1) // 2` |
| — | `--force` | Overwrite existing table | false |

---

## 5. Running XOR-SAT Sampling

Sample from the MAX-XOR-SAT distribution for one (m, rhs-index) pair and append one
JSONL line to an output file.

> **Block strategy:** the XOR-SAT sampler uses the `"random"` strategy — at every Gibbs
> step a fresh uniformly-random subset of `--block-size` bits is chosen for joint
> resampling.  This corresponds to Algorithm A1 of the paper.

```bash
# Single run: m=250, rhs-index=0
bin/xorsat-sampling -m 250 -i 0 -N 100000 -l 3 -o results_m250.jsonl

# Sweep all 100 RHS indices
for i in $(seq 0 99); do
    bin/xorsat-sampling -m 250 -i $i -N 100000 -l 3 -o results_m250.jsonl
done
```

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-m` | `--m` | Problem size | required |
| `-i` | `--rhs-index` | RHS index (0–99) | required |
| `-N` | `--num-samples` | Gibbs steps | required |
| `-l` | `--ell` | Polynomial degree | required |
| — | `--block-size` | Bits updated per step | 10 |
| — | `--num-burn-in` | Burn-in steps | 0 |
| `-s` | `--seed-offset` | Offset added to per-instance seed | 0 |
| `-o` | `--output` | JSONL output file (appended) | required |

---

## 6. Running XOR-SAT Resampling

Collect τ\_DQI values for one (m, rhs-index, algorithm) combination and append a JSONL
line to an output file.  The script implements the same two algorithms as OPI resampling
but for the MAX-XOR-SAT distribution.

> **Block strategy:** the XOR-SAT resampler uses the `"random"` strategy — at every Gibbs
> step a fresh uniformly-random subset of `--block-size` bits is chosen for joint
> resampling.

A "good sample" is a state x where n\_satisfied(x) / m ≥ predicted\_fraction (the
DQI-predicted fraction of satisfied constraints).

### Algorithm 1 (Continuous)

Run from a single seeded random start and record τ\_DQI at the first step where the
predicted threshold is reached.

```bash
for i in $(seq 0 99); do
    bin/xorsat-resampling -m 250 -i $i -a 1 -l 12 -o results_m250_alg1.jsonl
done
```

### Algorithm 2 (Restart)

Collect N good samples; each attempt uses an independent seed (`base_seed + attempt + 1`).

```bash
for i in $(seq 0 99); do
    bin/xorsat-resampling -m 250 -i $i -a 2 -l 12 -g 10 -o results_m250_alg2.jsonl
done
```

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-m` | `--m` | Problem size (number of constraints) | required |
| `-i` | `--rhs-index` | RHS index (0–99) | required |
| `-a` | `--algorithm` | 1 = Continuous, 2 = Restart | required |
| `-l` | `--ell` | Polynomial degree | required |
| — | `--block-size` | Gibbs block size | 3 |
| `-g` | `--num-good-samples` | (Alg 2) good samples to collect | 10 |
| `-s` | `--seed-offset` | Offset added to per-instance seed | 0 |
| `-x` | `--max-samples` | Safety limit: maximum Gibbs steps | 100 000 000 |
| `-o` | `--output` | JSONL output file (appended) | required |

The seed formula is `make_per_rhs_seed(p=2, rhs_idx, seed_offset)` — identical to the OPI
experiments.  For fully reproducible results set `PYTHONHASHSEED=0` in the environment
(or pass an explicit `--seed-offset`).

---

## 7. Running OPI Resampling

Collect τ\_DQI values for one (p, rhs-index, algorithm) combination and append JSONL
lines to an output file.

> **Block strategy:** the OPI sampler uses the `"permutation"` strategy — at the start
> of each epoch a fresh random permutation of all variables is drawn, then variables are
> updated in consecutive blocks of `--num-bit-flips` until every variable has been
> visited exactly once.  This corresponds to Algorithm A2 of the paper.

### Algorithm 1 (Continuous)

Keep sampling continuously until N unique "good samples" are collected.

```bash
for i in $(seq 0 9); do

    for p in 13 17 19 23 29 31 37 41 43 47; do
        bin/opi-resampling -p $p -i $i -a 1 -g 10 -o results_alg1_p${p}.jsonl
    done
done
```

### Algorithm 2 (Restart)

For each of N iterations, restart from a fresh random state until one good sample is found.

```bash
for i in $(seq 0 9); do
    for p in 13 17 19 23 29 31 37 41 43 47; do
        bin/opi-resampling -p $p -i $i -a 2 -g 10 -o results_alg2_p${p}.jsonl
    done
done
```

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-p` | `--p` | Prime modulus | required |
| `-i` | `--rhs-index` | RHS index (0–9) | required |
| `-a` | `--algorithm` | 1 = Continuous, 2 = Restart | required |
| — | `--num-bit-flips` | Gibbs block size | 10 |
| `-g` | `--num-good-samples` | Target good samples per run | 10 |
| `-s` | `--seed-offset` | Offset added to per-instance seed | 0 |
| `-x` | `--max-samples-per-attempt` | Safety limit per attempt | 10 000 000 |
| `-F` | `--n-factor` | n = p // n_factor | 2 |
| `-o` | `--output` | JSONL output file (appended) | required |

A "good sample" is a state x where f(x) > ⌊n\_predicted · m⌋, where n\_predicted is the
DQI-predicted fraction of satisfied constraints.

---

## 8. Inspecting Trajectories (OPI)

`bin/verify-opi-trajectories` runs a single Gibbs batch for one (p, rhs) pair and emits
a JSONL line containing the full new-best trajectory and the step at which predN is first
surpassed. Useful for quickly checking mixing behaviour before committing to a full sweep.

```bash
# Print one trajectory line to stdout
bin/verify-opi-trajectories -p 13 -i 0

# Custom batch size, append to file
bin/verify-opi-trajectories -p 13 -i 0 -b 1000000 -o trajectories_p13.jsonl
```

**Full option reference:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-p` | `--p` | Prime modulus | required |
| `-i` | `--rhs` | RHS index (0-based) | required |
| — | `--block-size` | Variables updated per step (max 3) | 3 |
| `-b` | `--batch-size` | Gibbs steps per batch | 500 000 |
| `-s` | `--seed-offset` | Offset added to per-instance seed | 0 |
| `-F` | `--n-factor` | n = p // n_factor | 2 |
| `-o` | `--output` | Append to file; omit to write to stdout | — |

---

## 9. Output Format

Both sampling scripts append **JSONL** (one JSON object per line) to the output file.

### XOR-SAT sampling output line (`xorsat-sampling`)

```json
{
  "m": 250, "v_index": 0, "n": 156, "num_samples": 100000,
  "ell": 3, "block_size": 10, "seed": 42, "seed_offset": 0,
  "max_fx": 198, "mean_fx": 176.3,
  "fx_values": [174, 176, ...]
}
```

### XOR-SAT resampling output — metadata line (written once per new file)

```json
{
  "type": "metadata", "ell": 4, "num_variables": 62, "num_constraints": 100,
  "predicted_fraction": 0.6413, "num_bit_flips": 3,
  "num_rhs": 100, "max_samples": 1e14,
  "rhs_indices": [78, 80, ...]
}
```

The Algorithm 2 metadata line additionally contains `"num_good_samples_per_rhs": 10`.

### XOR-SAT resampling output — result line (one per RHS index)

**Algorithm 1** (one good sample per line):
```json
{
  "type": "rhs_data", "rhs_idx": 0, "algorithm": 1,
  "ell": 4, "block_size": 3, "seed": 12345, "seed_offset": 0,
  "tau_dqi": 52, "observables": {"fx": 34.0, "wt_x": 34, "wt_bx": 54},
  "trajectory": [[1, 58, 4.15e-05], [2, 61, 8.3e-05], [9, 65, 3.7e-04]]
}
```

**Algorithm 2** (list of `num_good_samples` observables per line):
```json
{
  "type": "rhs_data", "rhs_idx": 0, "algorithm": 2,
  "ell": 4, "block_size": 3, "seed": 12345, "seed_offset": 0,
  "observables": [{"fx": 30.0, "wt_x": 28, "wt_bx": 52}, ...],
  "trajectory": [[42, 65, 2.13e-03], ...]
}
```

Each `trajectory` entry is `[tau, n_satisfied, wall_time_seconds]` at every new-best step.

### OPI output — metadata line (written once per new file)

```json
{
  "type": "metadata", "p": 13, "n": 6, "r": 6,
  "num_constraints": 12, "predicted_fraction": 0.732,
  "predN": 8, "num_bit_flips": 10, "sampler_type": "BlockGibbsSampler",
  "algorithm": 1, "seed_formula": "hash((p, rhs_idx)) & 0x7FFFFFFF + seed_offset"
}
```

### OPI output — result line (one per good sample)

```json
{
  "rhs_idx": 0, "algorithm": 1, "tau_dqi": 312, "f_value": 8,
  "sample_number": 1, "x": [3, 1, 4, 1, 5],
  "cpu_time": 0.041, "wall_time": 0.042
}
```

---

## 10. Running Tests

```bash
cd paper-release
.venv/bin/python -m pytest
# or, if the venv is activated:
pytest
```

The test suite runs smoke checks (shape assertions), seed-consistency checks, and
golden-value regressions for both samplers and both resampling algorithms.

The two golden OPI resampling tests (`test_alg1_golden`, `test_alg2_golden`) require
`problems/opi/rhs_nsamples100_p7_r3.npy`. Generate it first with:

```bash
bin/generate-opi-rhs -p 7
```

`tests/test_xorsat_results.py` adds 10 tests for the XOR-SAT resampling path:

| Test | What it checks |
|------|----------------|
| `test_alg1_xorsat_golden_m100[0,1,2]` | τ\_DQI and `{fx, wt_x, wt_bx}` exactly match golden values for Algorithm 1 |
| `test_alg2_xorsat_golden_m100[0,1,2]` | Per-attempt observables match golden values for Algorithm 2 |
| `test_alg1_data_structure` | All `data/algorithm1/*.jsonl` have correct schema (metadata + rhs\_data rows) |
| `test_alg2_data_structure` | All `data/algorithm2/*.jsonl` have correct schema + N observables per RHS |
| `test_alg1_m100_predicted_fraction_consistent` | Stored `predicted_fraction` matches what the code computes from the problem matrix |
| `test_alg2_m100_predicted_fraction_consistent` | Same check for the algo2 file |

The six golden/consistency tests skip gracefully if the m=100 RHS file has not been
generated yet. Generate it first with:

```bash
bin/generate-xorsat-rhs -m 100
```

---

## 11. Complete Experiment Walkthrough

The following sequence reproduces all paper experiments from scratch.

```bash
# Step 1 — install
uv venv --python 3.13
uv pip install "numba>=0.61"   # only needed on Python 3.13
uv pip install -e ".[dev]"

# Step 2 — generate problem files

for p in 13 17 19 23 29 31 37 41 43 47; do
    bin/generate-opi-rhs -p $p
done
for m in 100 250 500 750 1000 3000; do
    bin/generate-xorsat-rhs -m $m
done

# Step 3 — pre-compute XOR-SAT lookup tables (one-time)
for m in 100 250 500 750 1000 3000; do
    bin/precompute-xorsat-lookup -m $m
done

# Step 4 — run XOR-SAT sampling
for m in 100 250 500 750 1000 3000; do
    for i in $(seq 0 99); do
        bin/xorsat-sampling -m $m -i $i -N 100000 -l 3 -o results_xorsat_m${m}.jsonl
    done
done

# Step 5 — run XOR-SAT resampling (collect τ_DQI values, both algorithms)
# ell values: m=100→4, m=250→12, m=500→33, m=750→57, m=1000→86
declare -A ELL=(["100"]=4 ["250"]=12 ["500"]=33 ["750"]=57 ["1000"]=86)
for m in 100 250 500 750 1000; do
    l="${ELL[$m]}"
    for i in $(seq 0 99); do
        bin/xorsat-resampling -m $m -i $i -a 1 -l $l \
            -o results_xorsat_alg1_m${m}.jsonl
        bin/xorsat-resampling -m $m -i $i -a 2 -l $l -g 10 \
            -o results_xorsat_alg2_m${m}.jsonl
    done
done

# Step 6 — run OPI resampling (both algorithms)
for p in 13 17 19 23 29 31 37 41 43 47; do
    for i in $(seq 0 9); do
        bin/opi-resampling -p $p -i $i -a 1 -g 10 -o results_opi_alg1_p${p}.jsonl
        bin/opi-resampling -p $p -i $i -a 2 -g 10 -o results_opi_alg2_p${p}.jsonl
    done
done
```

---

## 12. Project Structure

```
paper-release/
├── bin/                               # Executable wrappers (no venv activation needed)
│   ├── generate-opi-rhs
│   ├── generate-xorsat-rhs
│   ├── precompute-xorsat-lookup
│   ├── xorsat-sampling
│   ├── xorsat-resampling              # NEW: XOR-SAT resampling (τ_DQI collection)
│   ├── opi-resampling
│   ├── sample-opi
│   ├── multisample-opi
│   └── verify-opi-trajectories
├── dqi_mcmc/                          # Python package
│   ├── __init__.py                    # Public API re-exports
│   ├── rng_state.py                   # RNG state utilities
│   └── api/
│       ├── _elementary_poly_fast.pyx  # Cython: elementary symmetric polynomials
│       ├── block_gibbs_sampler.py     # BlockGibbsSamplerOptimized (XOR-SAT)
│       ├── lookup_cache.py            # HDF5 lookup-table cache
│       ├── sat_problem_base.py        # Abstract problem base class
│       ├── maxxorsat/
│       │   ├── max_xor_p_sat_problem.py  # MaxXorPSatProblem + lookup table
│       │   └── xorsat_problem_reader.py  # Load LDPC matrix + RHS from disk
│       └── maxlinsat/
│           ├── block_gibbs_sampler.py    # BlockGibbsSampler (OPI)
│           └── max_opi_problem.py        # MaxOPIProblem + lookup table
├── scripts/
│   ├── generate_opi_rhs.py            # CLI: generate OPI RHS files
│   ├── generate_xorsat_rhs.py         # CLI: generate XOR-SAT RHS files
│   ├── precompute_xorsat_lookup.py    # CLI: pre-compute lookup tables
│   ├── run_opi_resampling.py          # CLI: OPI resampling (Alg 1 & 2)
│   ├── run_xorsat_sampling.py         # CLI: XOR-SAT Block Gibbs sampling
│   ├── run_xorsat_resampling.py       # CLI: XOR-SAT resampling (Alg 1 & 2)  NEW
│   ├── sample_opi.py                  # CLI: single OPI trial
│   ├── multisample_opi.py             # CLI: collect N good OPI samples
│   └── verify_opi_trajectories.py     # CLI: single-batch trajectory inspection
├── tests/
│   ├── conftest.py                    # Shared fixtures and markers
│   ├── test_xorsat_gibbs.py           # XOR-SAT smoke + golden tests
│   ├── test_xorsat_results.py         # XOR-SAT resampling golden + structural tests  NEW
│   ├── test_opi_gibbs.py              # OPI smoke + golden tests
│   └── test_opi_resampling.py         # OPI resampling smoke + golden tests
├── problems/
│   ├── maxxorsat/                     # LDPC matrices (committed)
│   └── opi/                           # OPI RHS files (generated by bin/generate-opi-rhs)
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Package metadata and tool config
├── requirements.txt                   # Pinned dependencies
└── setup.py                           # Cython extension build

data/                                  # Pre-computed paper results (repo root)
├── algorithm1/                        # Algorithm 1 (Continuous) XOR-SAT τ_DQI results
│   │                                  # 55 JSONL files, m ∈ {100,250,500,750,1000,1250,1500,1750}
│   └── gibbs3_m{M}_n{N}_ell{L}_num_rhs100_{batch}.jsonl
├── algorithm2/                        # Algorithm 2 (Restart) XOR-SAT results
│   │                                  # 5 JSONL files, m ∈ {100,250,500,750,1000}
│   └── gibbs3_m{M}_n{N}_ell{L}_num_rhs100_algo2.jsonl
├── maxxorsat/                         # Supporting XOR-SAT data
└── opi/                               # OPI experiment data
```

---

## 13. Citation

If you use this code, please cite:

```bibtex
@misc{gilfuster2026dqimcmc,
  author        = {Gil-Fuster, Elies and Ninio, Matan and Bittel, Lennart
                   and Shimoni, Yishai and Eisert, Jens and Woerner, Stefan
                   and {Carrera V\'{a}zquez}, Almudena},
  title         = {Approximate sampling from decoded quantum interferometry
                   via {Markov} chain {Monte Carlo} methods},
  year          = {2026},
  eprint        = {2607.28120},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2607.28120},
}
```
