# build-your-af3

[中文版 README](README.md)

An educational, Mac-friendly **AlphaFold 3** reimplementation. A
chapter-by-chapter Python package layout that loads the official
**ByteDance Protenix** checkpoints unchanged; runs on CPU, Apple-Silicon
MPS, or CUDA.

## Project layout

The project is organized into three top-level folders:

```
build-af3-from-scratch/
├── lessons/      # per-chapter writeups (markdown)
├── tutorials/    # fill-in-the-blank version (auto-generated)
└── solutions/    # complete reference implementation
    ├── attention/                # MHA + LayerNorm + AttentionPairBias
    ├── feature_extraction/       # JSON / MSA / template → tensors
    ├── feature_embedding/        # input embed + Atom Attention Encoder
    ├── pairformer/               # Pairformer + MSAModule + Triangle ops
    ├── diffusion/                # DiffusionModule + Transformer + sampler
    ├── confidence/               # ConfidenceHead + scoring
    ├── model/                    # top-level Protenix + inference driver
    ├── configs/                  # config dicts + ConfigDict parser (shared)
    └── runtime/                  # seed / logger / torch utils (shared)
```

Each chapter is a flat Python package; intra-chapter imports use the
fully-qualified form `from <chapter>.<file> import <Class>`.

## The three companion folders

| Folder | What's in it | How to use |
|---|---|---|
| `solutions/` | Complete reference implementation | Read it after you've done your own pass — try not to peek beforehand. |
| `tutorials/` | Auto-extracted from `solutions/` by `prepare_tutorials.py`. Every wrapped `forward` body is replaced with `pass`, and the TODO block above it carries detailed pseudocode. | Where you fill in the blanks. |
| `lessons/` | Per-chapter teaching markdown | For algorithm / math background. |

Regenerate the blanks:

```bash
python prepare_tutorials.py          # solutions/* → tutorials/*
python prepare_tutorials.py --clean  # wipe tutorials/ first
```

## How to use this project

Recommended 6-step path. Each chapter takes ~0.5–2 hours.

1. **Clone + generate the blanks**

   ```bash
   git clone https://github.com/Terry-Wang-Lynx/build-af3-from-scratch.git
   cd build-af3-from-scratch
   python prepare_tutorials.py --clean
   ```

   You'll work inside `tutorials/`. `solutions/` is the answer key —
   try not to peek beforehand.

2. **Set up the environment** (pick one)

   ```bash
   conda env create -f environment_mac.yml      # Mac / Apple Silicon
   # or
   conda env create -f environment_cpu.yml      # Linux / Mac CPU
   conda activate af3
   ```

3. **Walk through chapter notebooks.** Each
   `tutorials/<chapter>/<chapter>.ipynb` is a guided lab: every section
   tells you which `.py` to open and which TODOs to fill, followed by a
   test cell:

   ```python
   test_module_shape(mha, 'mha_gated', control_folder)
   test_module_forward(mha, 'mha_gated', inputs=(...), ...)
   ```

   - Passes → continue.
   - Fails → the test helper prints which output mismatched. Go back to
     the `.py`, fix, re-run the cell (autoreload picks it up; no kernel
     restart needed).

   Recommended order:

   | Step | Chapter | Notebook | What you build |
   |---|---|---|---|
   | 0 | `feature_extraction/` | `feature_extraction.ipynb` | **Read-only** tour: JSON → feature dict |
   | 1 | `attention/` | `attention.ipynb` | Linear / LayerNorm / MHA / AdaLN / Transition / AttentionPairBias |
   | 2 | `pairformer/` | `pairformer.ipynb` | OuterProductMean / TriangleMul / TriangleAttention / MSAPairWeightedAveraging / PairformerBlock |
   | 3 | `feature_embedding/` | `feature_embedding.ipynb` | RelativePositionEncoding / FourierEmbedding |
   | 4 | `diffusion/` | `diffusion.ipynb` | ConditionedTransitionBlock / DiffusionTransformerBlock / DiffusionTransformer / expressCoordinatesInFrame / centre_random_augmentation |
   | 5 | `confidence/` | `confidence.ipynb` | DistogramHead |
   | 6 | `model/` | `overview.ipynb` | End-to-end: load Protenix weights, run inference, write CIF |

4. **Sanity check after each chapter**:
   ```bash
   python generate_control_values.py --verify --src tutorials --chapters <chapter>
   ```

   To run every notebook in one shot:
   ```bash
   python check_solutions.py --src tutorials
   ```

5. **Run end-to-end.** Once every blank is filled in, download the
   Protenix checkpoint (see "Quickstart") and open
   `tutorials/model/overview.ipynb` to run a forward pass; you'll get a
   `7r6r_pred.cif` plus pLDDT / pTM scores.

6. **Compare with the reference.** Diff your finished `tutorials/`
   against `solutions/` to spot any rougher edges in your version.
   Inference output should match (state_dict is fully compatible).

## Per-module unit tests (control values)

Each chapter ships a small set of reference tensors under
`control_values/`. Together with `runtime/checks.py`
(`test_module_shape`, `test_module_forward`) they let students validate
each function on its own:

- Module parameters are temporarily replaced with
  `torch.linspace(-1, 1, numel)`. The module is then run on a fixed
  `test_inputs` and the output is compared to `<name>_out.pt`.
  Parameter shapes are compared to `<name>_param_shapes.pt`.
- `solutions/<chapter>/control_values/_generate.py` both writes the
  reference files (`overwrite=True`) and verifies against them
  (`overwrite=False`).
- A single top-level entry point at the repo root:

```bash
# Regenerate every reference .pt from solutions/.
python generate_control_values.py

# Verify your tutorials/ implementation (will fail until filled in).
python generate_control_values.py --verify --src tutorials
```

The `.pt` files are tiny (KB-sized) and committed alongside the source.
`prepare_tutorials.py` copies the whole `control_values/` tree into
`tutorials/`, so a fresh clone can run function-level tests immediately.

## Quickstart

### 1. Install

Mac / Apple Silicon:

```bash
conda env create -f environment_mac.yml
conda activate af3
```

CPU only (Linux / Mac):

```bash
conda env create -f environment_cpu.yml
conda activate af3
```

Or venv + pip:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision \
            rdkit biopython biotite modelcif gemmi pdbeccdutils \
            ml-collections scipy pandas scikit-learn scikit-learn-extra \
            matplotlib ipykernel ipywidgets py3dmol icecream fair-esm
```

### 2. Download a checkpoint

We use the official Protenix-Tiny (~110 M params, MSA-based):

```bash
mkdir -p checkpoints
curl -L -o checkpoints/protenix_tiny_default_v0.5.0.pt \
    https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt
```

Auxiliary caches (CCD chemistry, etc.) auto-download to `~/common/` on
first run.

### 3. Run inference

```bash
cd solutions
LAYERNORM_TYPE=torch python -m model.inference \
    --input_json examples/example.json \
    --dump_dir   ./out \
    --device     mps            # or cpu / cuda
```

For each sample the runner produces a `*.cif` and a
`*_summary_confidence_*.json`. A complete example JSON ships at
`solutions/examples/example.json` (bundled 7r6r protein + MSA).

### 4. End-to-end demo

```bash
jupyter notebook solutions/overview.ipynb
```

The notebook walks through: build model → load weights → featurize → run
inference → write CIF → visualize.

### 5. Verify

A ~200-residue protein takes ~4–10 s per forward on CPU with pLDDT in
the 30–75 range (depends on PyTorch version + backend); MPS is ~1.6×
faster.

## How the chapters compose

```
                       ┌─ feature_extraction ─┐
                       │  JSON → atom array   │
                       │  MSA / templates     │
                       └──────┬───────────────┘
                              ▼
       ┌────────────────────────────────────────────┐
       │  feature_embedding                         │
       │  · InputFeatureEmbedder (Alg 2)            │
       │  · AtomAttentionEncoder (Alg 5)            │
       │  · RelativePositionEncoding                │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  pairformer  (Alg 8/16/17)                 │
       │  · MSAModule · TemplateEmbedder            │
       │  · PairformerStack with Triangle ops       │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  diffusion  (Alg 18/23/25)                 │
       │  · DiffusionModule + Transformer           │
       │  · Sampler (InferenceNoiseScheduler)       │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  confidence  (Alg 26–31)                   │
       │  · ConfidenceHead + DistogramHead          │
       │  · pTM / iPTM / pLDDT / clash              │
       └────────────────────────────────────────────┘
```

`model/model.py` wires it all up following Algorithm 1.

## Features

- Chapter-by-chapter Python package layout — every block lives in the file
  named after it.
- Pure-PyTorch model: runs on CPU, Apple-Silicon MPS, and CUDA.
- Loads the official ByteDance Protenix Tiny / Mini checkpoints out of the box.
- Bilingual (English + 中文) docstrings on the public API.
- Every wrapped `forward` / key `__init__` ships with a detailed-pseudocode
  TODO so filling in the blanks reduces to mechanical transcription — and
  the result is bit-for-bit compatible with the Protenix checkpoint.

## Acknowledgements

- **ByteDance Protenix** — AF3 architecture implementation and open weights.
- **`alphafold-decoded`** by Kilian Mandon (AF2 educational project) — the
  inspiration for the chapter-by-chapter pedagogical layout.
- **DeepMind** — AlphaFold 3 paper.

## License

Apache 2.0. See [LICENSE](LICENSE).
