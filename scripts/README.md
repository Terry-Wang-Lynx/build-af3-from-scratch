# scripts/

Maintainer-only helpers. Students should **not** need to run any of these.

| Script | Purpose |
|---|---|
| `build_chapter_notebooks.py` | Regenerate `solutions/<chapter>/<chapter>.ipynb` for the five chapter labs. Run after editing the cell templates inside the script. |
| `build_feature_extraction_notebook.py` | Regenerate `solutions/feature_extraction/feature_extraction.ipynb` (the read-only walkthrough). |
| `build_overview_notebook.py` | Regenerate `solutions/model/overview.ipynb` (the end-to-end demo that auto-detects whether it sits under `solutions/` or `tutorials/`). |

All scripts assume they are invoked from the repository root:

```bash
python scripts/build_chapter_notebooks.py
python scripts/build_feature_extraction_notebook.py
python scripts/build_overview_notebook.py
```

Both rewrite the `.ipynb` files in place. After running, the next
`python prepare_tutorials.py` call propagates the updated notebooks
into `tutorials/`.
