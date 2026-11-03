# Hồn Việt: Ca Dao Search & Q&A

A Flask web app for searching and learning Vietnamese folk literature (ca dao, tục ngữ, thành ngữ). About 8,600 entries were scanned from the two-volume dictionary *Từ điển Tục ngữ Thành ngữ Ca dao Việt Nam* (Quyển Thượng and Quyển Hạ). The entries were OCR'd, reviewed by hand and enriched with an LLM, then indexed for semantic search.

The site has these pages:

| Page | URL | What it does | Needs |
|---|---|---|---|
| Search | `/` | Semantic and keyword search over all entries, with a category filter | `embeddings.pkl`, `keywords.pkl`, ML libraries |
| Ca dao vấn đáp (Q&A) | `/qa-page` | The user describes a worry. The app finds the 3 best-matching sayings, reranks them, and asks OpenAI to pick one and write advice | `advice_index.pkl`, `CA_DAO_VAN_DAP_KEY`, ML libraries |
| Wiki | `/wiki-page` | Browse every entry A–Z, with the cropped scan image of the original page | `data/wiki.db` (in git), `extractions/output-folder/*.png` |
| Learning | `/learning-page` | A fill-in-the-blank game over curated poems, with likes/dislikes and user interpretations | `data/learning-data.json` |

The nav bar only shows **Search** and **Q&A**. Wiki and Learning still work, but you reach them by typing the URL. The UI is in Vietnamese and English; the language is chosen by the `lang` cookie (strings are in `i18n.py`).

---

## Quick start: run the website

If someone has already given you the data files, this is all you need.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                 # then fill in the keys (see "Environment variables")

make dev                             # Flask dev server (debug) on http://localhost:4000
# or
make prod                            # waitress in the background, logs in ./log.txt
make status | make logs | make stop | make restart
```

Kill whatever is holding the port with `fuser -k 4000/tcp` (Linux) or `lsof -ti :4000 | xargs kill` (macOS).

On first start the server downloads two Hugging Face models: `AITeamVN/Vietnamese_Embedding` (BGE-M3 based) and `AITeamVN/Vietnamese_Reranker`. They take several GB of disk and RAM, so expect the first boot to be slow. No GPU is required.

**The server degrades gracefully.** If `torch`, `faiss` or `sentence-transformers` is not installed, or the `.pkl` files are missing, the server still starts. Search then returns "no data", Q&A falls back to a random poem, and Wiki and Learning work as normal. This lets you work on the frontend on a lightweight machine.

### Dependencies

- `requirements.txt` holds what the server and indexing steps need. It was frozen on a Linux CUDA machine; the CUDA-only packages (`nvidia-*`, `cuda-*`, `triton`) are marked `sys_platform == "linux"`, so pip skips them on macOS.
- `requirements-ocr.txt` holds the extra packages for the OCR pipeline (Steps 1–2): PyMuPDF, OpenCV and PyObjC. PyObjC is macOS only.

---

## Data you need

Almost none of the data is in git. `.gitignore` excludes `*.csv`, `*.pkl`, `*.json`, `*.pdf` and `*.txt`, so a fresh clone has only the code, `data/wiki.db` and `data/user_interpretations.txt`. Get the rest from the team's shared storage and put it at these exact paths:

| File | Size / rows | Used by | How it is made |
|---|---|---|---|
| `embeddings.pkl` (repo root) | ~8,200 poems × 3 vectors | Search | Step 5: `step2_index.py` |
| `keywords.pkl` (repo root) | ~76 keywords | Search | Step 5: `step2_index.py` |
| `advice_index.pkl` (repo root) | ~690 poems | Q&A | Step 6: `advice/advice_index.py` |
| `advice/advice.csv` (or `./advice.csv`) | 687 rows | Q&A fallback only | Curated subset of `wiki.csv`; how it was made is not recorded in the repo |
| `data/learning-data.json` | 33 poems | Learning page | Written by hand; edit it with the tools in [Learning data](#learning-data) |
| `data/wiki.db` | 8,651 rows, **in git** | Wiki, plus poem IDs and explanations for Search and Q&A | Built automatically from `extractions/wiki.csv` |
| `extractions/output-folder/<id>.png` | ~8,700 images | Wiki scan images | Step 2: OCR editor |
| `extractions/wiki.csv` | 8,651 rows | Step 5; rebuilding `wiki.db` | Step 4 |

You only need the first group (the `.pkl` files, `advice.csv` and `learning-data.json`) to *run* the site. The other files are only needed to *rebuild* the data.

### Main CSV schemas

- `extractions/wiki.csv` is the master dataset: `id, Alphabet, poem, category, explanation, pages, searchable_explanation, keywords`
  - `id` is a UUID. It is also the image name in `output-folder/`.
  - `category` is one of `(CA DAO)`, `(TỤC NGỮ)`, `(THÀNH NGỮ)`, and so on.
  - `explanation` is the original dictionary text.
  - `searchable_explanation` is a 2–3 sentence LLM summary, used for embedding.
  - `keywords` is a comma-separated list drawn from `extractions/keywords.json`.
- `advice/advice.csv`: `id, poem, category, searchable_explanation, keywords`
- `filter/input.csv` → `filter/filter-dataset.csv`: `id, poem, category, explanation, Quyển[, keyword, type]`

---

## The data pipeline

```
 1.pdf, 2.pdf  (scanned dictionary, 2 volumes)
     │  Step 1: extractions/step1.py, extractions/cut.py    split 2-page spreads, drop junk pages
     ▼
 1-split.pdf, 2-split.pdf
     │  Step 2: extractions/step2_improved.py               OCR editor (Apple Vision, macOS only)
     ▼
 poem-extraction.csv  +  output-folder/<id>.png
     │  (both volumes merged by hand, "Quyển" column added)
     ▼
 filter/input.csv
     │  Step 3: filter/review_app.py                        human review / cleanup UI
     ▼
 filter/filter-dataset.csv
     │  Step 4: extractions/step3_metadata.py               OpenAI adds searchable_explanation + keywords
     ▼
 extractions/wiki.csv  ──────────────────────────────►  data/wiki.db (auto-built by the server)
     │  Step 5: step2_index.py
     ▼
 embeddings.pkl, keywords.pkl                         ──►  Search

 advice/advice.csv
     │  Step 6: advice/advice_index.py
     ▼
 advice_index.pkl                                     ──►  Q&A
```

**Most of the time you only need Step 5 and/or Step 6**, after editing `wiki.csv` or `advice.csv`. Steps 1–4 have already been run and their outputs are in shared storage. Re-running them means redoing OCR and manual review of thousands of entries.

The step numbers in this README don't match the file names. `step1_extract.py` at the root is a legacy script, and the root `step2_index.py` is the real Step 5. See [What's legacy](#whats-legacy--safe-to-ignore).

### Step 1: Prepare the PDFs

```bash
cd extractions
python step1.py                                   # 2.pdf → 2-split.pdf (splits each 2-page spread into single pages)
python cut.py <pdf> 1,2,51-53                     # remove covers/blank pages (1-based); writes <name>_cut.pdf
```

`step1.py` has its input (`2.pdf`) and output hard-coded at the bottom; edit the paths there to process volume 1. Install their dependencies with `pip install -r requirements-ocr.txt`.

### Step 2: OCR the scans into CSV (macOS only)

```bash
pip install -r requirements-ocr.txt
cd extractions
python step2_improved.py            # opens the OCR editor on http://127.0.0.1:5001
```

This is a web tool that uses Apple's Vision framework for Vietnamese OCR. You pick a `*-split.pdf`, adjust the boxes around each entry, and export. Exporting writes `poem-extraction.csv` and crops each entry into `output-folder/<id>.png`. Page edits are kept **in memory only**, so restarting the tool loses unsaved work.

`extractions/step2.py` is the older non-interactive batch version (`2-split.pdf` → `2-extracted-poems.csv`). The root `apple_ocr.py` is a general-purpose OCR command-line tool (`python apple_ocr.py file.pdf -o out.txt`) and is not part of the pipeline.

### Step 3: Review and clean entries by hand

```bash
python filter/review_app.py         # http://localhost:4005
```

Reads `filter/input.csv`, lets a reviewer step through and fix each entry, and writes `filter/filter-dataset.csv`. Progress is saved in `filter/review_state.json`, so you can stop and resume.

### Step 4: Add LLM metadata (searchable explanation + keywords)

Needs `CHATGPT_KEY` in `.env`. Uses `gpt-5-mini` for the text and `text-embedding-3-small` to merge near-duplicate keywords.

```bash
cd extractions                      # must run here: it reads/writes keywords.json in the current dir
python step3_metadata.py --input <your input>.csv --output wiki.csv --workers 20
```

- `--input` and `--output` are required. There are no defaults, so a careless re-run can't regenerate (and pay for) the whole dataset.
- It is resumable: rows that already have both fields are skipped, and progress is saved periodically. To continue an interrupted run, point `--input` at the partial output file.
- Keywords are restricted to the list in `keywords.json`. New keywords are only added when nothing in the list is ≥ 0.85 similar. Embeddings are cached in `.keywords_embeddings.json`.
- This step costs OpenAI money (~8,600 rows).

### Step 5: Build the search index

```bash
# run from the repo root
python step2_index.py
python check.py                      # optional sanity check of the .pkl files
```

Reads `extractions/wiki.csv` and writes `embeddings.pkl` and `keywords.pkl` to the **current directory**. The server reads them from the repo root, so run it from the root. Rows with no `searchable_explanation` or `keywords` are skipped. Each poem gets 3 vectors: poem text, meaning and keywords. Runs locally; there is no API cost.

After it finishes, restart the server.

### Step 6: Build the Q&A index

```bash
python advice/advice_index.py       # reads advice/advice.csv, writes ./advice_index.pkl at the repo root
```

Restart the server afterwards.

### Rebuilding `data/wiki.db`

On startup the server imports `extractions/wiki.csv` into `data/wiki.db`, but only if the table is empty. If you change `wiki.csv`, delete `data/wiki.db` and restart the server to rebuild it. The database is committed to git, so commit the new one too.

---

## Learning data

The Learning page reads a JSON list of `{topic, poems: [{poem_text, blanks: [{word, options}], introduction, interpretation}]}`.

Two ways to edit it:

```bash
# JSON <-> spreadsheet-friendly CSV
python data/convert.py data/learning-data.json data/final.csv           # export
python data/convert.py data/final.csv data/learning-data.json --reverse # import

# JSON <-> plain text (scripts/learning_data_editor.txt)
cd scripts
python edit_learning_data.py --out    # JSON -> learning_data_editor.txt
python edit_learning_data.py --in     # learning_data_editor.txt -> JSON
```

User likes and dislikes are appended to `data/user_interpretations.txt` as `POEM_ID | USERNAME | STATUS | TEXT`. This file is tracked in git and holds real user input, so be careful about what you commit.

---

## Environment variables (`.env`)

| Variable | Used by | Purpose |
|---|---|---|
| `CA_DAO_VAN_DAP_KEY` | `step3_server.py` | OpenAI key for the Q&A page (model `gpt-5.1`). Without it, Q&A returns the top match's stock explanation and no generated advice. |
| `CHATGPT_KEY` | `extractions/step3_metadata.py` | OpenAI key for Step 4 |
| `APOLLO_API_KEY` | `step1_extract.py` (legacy) | Remote Ollama server; not needed |

`.env.example` only lists `CHATGPT_KEY`; add the others yourself.

---

## How search works (`step3_server.py`)

1. Embed the query and search the multi-vector FAISS index (cosine ≥ 0.6).
2. Find the closest keywords (≥ 0.6) and add every poem tagged with them.
3. If there are still no results, relax the threshold to 0.25, then force-accept the top 3 keywords.
4. As a last resort, do a plain substring search.
5. Sort by score, apply the category filter, and look up each poem's `id` in `wiki.db`.

Q&A works like this: embed the concern, take the max score per poem over its 3 vectors, keep the top 20, rerank them with the CrossEncoder, keep the top 3, then let OpenAI pick one and write the advice as JSON.

The thresholds are constants at the top of `step3_server.py`.

---

## Repo map

```
step3_server.py          ★ the web app (Flask). Everything the site does is here.
step2_index.py           ★ Step 5: build embeddings.pkl / keywords.pkl
advice/advice_index.py   ★ Step 6: build advice_index.pkl
i18n.py                    UI strings (vi / en)
templates/, static/        Frontend for the main site
Makefile                   dev / prod / stop / status / logs
check.py                   Sanity-check the .pkl files

extractions/               Steps 1, 2 and 4 (PDF → CSV → enriched CSV), plus the OCR editor's own templates_editor/ and static/
filter/                    Step 3 review app (own templates/ and static/)
data/                      wiki.db, learning data, user interpretations, convert.py
scripts/                   Learning-data text editor

step1_extract.py           LEGACY: old Ollama/sailor2 meaning extractor
filter.py                  LEGACY: one-off line-joining cleanup of an old input.csv
apple_ocr.py               Standalone OCR command-line tool, not in the pipeline
scratch/                   One-off debugging scripts
OllamaServerUserGuide_*.html   Docs for the remote Ollama server used by the legacy script
instruction.md, .agents/AGENTS.md   Rules for AI coding assistants
```

### What's legacy / safe to ignore

- **`step1_extract.py`**: the first version of the enrichment step (Ollama `sailor2:20b` on a remote server, `input.csv` → `output.csv` with `Meaning`/`Keywords` columns). It was replaced by `extractions/step3_metadata.py`. Note that it calls `main()` at import time.
- **`filter.py`**, **`extractions/step2.py`**, **`extractions/1-extracted-poems.csv`, `2-extracted-poems.csv`**: superseded by the OCR editor and the review app.
- **`embedding.log`**, **`extractions/scratch/`**, **`.DS_Store`** files: leftover artifacts that were committed by accident.

---

## Gotchas

1. **Run scripts from the right directory.** Paths are relative to the current working directory:
   - `step2_index.py` and `check.py`: the repo root.
   - `step3_metadata.py`: `extractions/`.
   - `edit_learning_data.py`: `scripts/`.
2. **The Makefile has two virtualenvs.** `make dev` and `make prod` use `./venv`, while `make dev-local` uses `./.venv`.
3. **`/learning-page` and `/wiki-page` are not in the nav bar** (only Search and Q&A are). This looks intentional after the redesign in commit `38accc0`.

---

## Notes for AI coding assistants

See `.agents/AGENTS.md` and `instruction.md`. In short: the data files (`*.csv`, `*.pkl`) may not exist on the machine you are editing, so don't search for them or create mock data. Don't start the server or test the website unless you are explicitly asked to. Never read `.env`.
