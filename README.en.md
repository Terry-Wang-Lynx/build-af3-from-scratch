# build-your-af3

[中文版 README](README.md)

An educational, Mac-friendly **AlphaFold 3** reimplementation, built in the
same chapter-by-chapter format as
[alphafold-decoded](https://github.com/kilianmandon/alphafold-decoded) (AF2)
but targeting AF3 architecture and weights. Loads the official **ByteDance
Protenix** checkpoints unchanged; runs on CPU, Apple-Silicon MPS, or CUDA.

## Project layout

Mirrors the reference's three-fold structure:

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
fully-qualified form `from <chapter>.<file> import <Class>`, identical to
the AF2 reference.

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

## Why an AF3 version?

AF3 introduces architectural changes that don't fit the AF2 mold:

| AF2 chapter           | AF3 equivalent                            |
|-----------------------|-------------------------------------------|
| `evoformer`           | `pairformer` (+ `msa_stack`)              |
| `structure_module`    | `diffusion`                               |
| `feature_embedding`   | + AtomAttentionEncoder                    |
| `geometry`            | mostly absorbed into the diffusion module |

The middle chapters therefore differ from AF2, but the low-level primitives
(attention / residual / axial ops) are the same.

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
- **`alphafold-decoded`** by Kilian Mandon — the chapter-by-chapter
  pedagogical format.
- **DeepMind** — AlphaFold 3 paper.

## License

Apache 2.0. See [LICENSE](LICENSE).
