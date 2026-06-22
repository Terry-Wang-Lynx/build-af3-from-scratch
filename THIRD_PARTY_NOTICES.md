# Third-Party Notices

This repository (`build-af3-from-scratch`) is an educational AlphaFold 3 /
Protenix inference reimplementation. It is distributed under the Apache License
2.0 (see [`LICENSE`](LICENSE)). Portions of the code and the project structure
are derived from, or adapted from, the third-party projects listed below. Their
copyright notices and license terms are reproduced here in full so that
downstream users have the complete provenance and license terms without needing
to follow external links.

---

## 1. ByteDance Protenix

- **Component**: Model architecture, configuration files, trained-weight
  loaders, feature pipeline, and most modules under `solutions/` and
  `tutorials/`.
- **Source**: https://github.com/bytedance/Protenix
- **Copyright**: Copyright 2024 ByteDance and/or its affiliates.
- **License**: Apache License, Version 2.0.

These files carry the standard Apache 2.0 per-file header
(`# Copyright 2024 ByteDance and/or its affiliates.`). The full Apache License
2.0 text is reproduced in [`LICENSE`](LICENSE) and applies to this repository
as a whole.

---

## 2. AlQuraishi Laboratory / OpenFold-derived utilities

- **Component**: A small number of utility files (e.g. template parsing and
  data-pipeline helpers) that ByteDance Protenix itself adapted from
  OpenFold. These files carry an additional
  `# Copyright 2021 AlQuraishi Laboratory` header alongside the ByteDance
  Apache 2.0 header.
- **Source**: https://github.com/aqlaboratory/openfold
- **Copyright**: Copyright 2021 AlQuraishi Laboratory.
- **License**: Apache License, Version 2.0.

Files currently bearing this header (run `grep -rl AlQuraishi solutions` for
the authoritative list):

- `solutions/feature_extraction/template/template_parser.py`
- `solutions/pairformer/triangle_ops.py`
- `solutions/pairformer/triangle.py`

The same headers are mirrored in the corresponding `tutorials/` files.

The full Apache License 2.0 text is reproduced in [`LICENSE`](LICENSE).

---

## 3. alphafold-decoded (pedagogical scaffold)

- **Component**: The pedagogical structure — chapter layout, per-chapter
  exercise scaffolding, the control-value testing approach, and
  `prepare_tutorials.py` (which strips reference `solutions/` into the
  student-facing `tutorials/` fill-in version).
- **Source**: https://github.com/kilianmandon/alphafold-decoded
- **Copyright**: Copyright (c) 2024 Kilian Mandon.
- **License**: MIT License (reproduced in full below).

### MIT License

```
MIT License

Copyright (c) 2024 Kilian Mandon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 4. Reference materials (not redistributed)

The teaching text under `lessons/` is original writing. It references, but
does not reproduce, the following copyrighted works:

- **AlphaFold 3 paper** — Abramson et al., *Accurate structure prediction of
  biomolecular interactions with AlphaFold 3*, Nature (2024). Figures and
  algorithms are referenced by number only; no figures are embedded.
- **"The Illustrated AlphaFold"** — Elana Pearl Simon (2024),
  https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/ — credited
  as structural inspiration for the chapter division. No text or figures from
  the blog are reproduced.

---

> Note on scope: this file lists the third-party *source code* and *project
> structure* incorporated into this repository. Runtime Python dependencies
> (PyTorch, RDKit, Biotite, etc.) are installed separately via pip/conda and
> retain their own respective licenses; they are not redistributed here.
