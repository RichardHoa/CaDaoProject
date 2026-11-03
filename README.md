# Hồn Việt: Ca Dao Search & Q&A

A Flask web app for searching and learning Vietnamese folk literature (ca dao, tục ngữ, thành ngữ). About 8,600 entries were scanned from the two-volume dictionary *Từ điển Tục ngữ Thành ngữ Ca dao Việt Nam* (Quyển Thượng and Quyển Hạ). The entries were OCR'd, reviewed by hand and enriched with an LLM, then indexed for semantic search. Domain terms are defined in [`GLOSSARY.md`](GLOSSARY.md).

The site has these pages:

| Page | URL | What it does | Needs |
|---|---|---|---|
| Search | `/` | Semantic and keyword search over all entries, with a category filter | `data/indexes/embeddings.pkl`, `data/indexes/keywords.pkl`, ML libraries |
| Ca dao vấn đáp (Q&A) | `/qa-page` | The user describes a worry. The app finds the 3 best-matching entries, reranks them, and asks OpenAI to pick one and write advice | `data/indexes/advice_index.pkl`, `data/advice/advice.csv`, `CA_DAO_VAN_DAP_KEY`, ML libraries |
| Wiki | `/wiki-page` | Browse every entry A–Z, with the cropped scan image of the original page | `data/wiki/wiki.csv` (in git), `data/images/*.png` |
| Learning | `/learning-page` | A fill-in-the-blank game over curated poems, with likes/dislikes and user interpretations | `data/learning/learning-data.json` (in git) |

The nav bar only shows **Search** and **Q&A**. Wiki and Learning still work, but you reach them by typing the URL. The UI is in Vietnamese and English; the language is chosen by the `lang` cookie (strings are in `app/i18n.py`).

---

## Quick start: run the website

This gets the server running. Search, Q&A and the Wiki scan images also need data files that aren't in git; see [Data you need](#data-you-need).

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

**The server degrades gracefully.** If `torch`, `faiss` or `sentence-transformers` is not installed, or the index files are missing, the server still starts. Search then returns "no data", Q&A falls back to a random entry, and Wiki and Learning work as normal. This lets you work on the frontend on a lightweight machine.

### Run everything from the repo root

Every script is a module run with `python -m` from the repo root, for example `python -m app.server` or `python -m pipeline.build_search_index`. All data paths are defined once in [`paths.py`](paths.py) and resolve from the repo root.

### Dependencies

- `requirements.txt` holds what the server and indexing steps need. It was frozen on a Linux CUDA machine; the CUDA-only packages (`nvidia-*`, `cuda-*`, `triton`) are marked `sys_platform == "linux"`, so pip skips them on macOS.
- `requirements-ocr.txt` holds the extra packages for the OCR pipeline (Steps 1–2) and the OCR tools: PyMuPDF, OpenCV and PyObjC. PyObjC is macOS only.

---

## Data you need

The code is in git, but most of the data the site runs on is not: it is either too large or generated, and `.gitignore` excludes most of `data/`. **You do not need to build any of it yourself.** The finished files already exist; you copy them from the team's shared storage into the right folders.

### 1. What you already have after `git clone`

| File | What it is |
|---|---|
| `data/wiki/wiki.csv` | The master dataset: every entry (8,651 rows) with its explanation and keywords. On first start the server builds `data/wiki/wiki.db` from it, which powers the Wiki page and gives Search and Q&A their entry IDs. |
| `data/learning/learning-data.json` | The 33 curated poems for the Learning game. See [Learning data](#learning-data) to edit it. |
| `data/learning/user_interpretations.txt` | Likes/dislikes that visitors leave on the Learning page. |
| `data/enrichment/keywords.json` | The controlled keyword vocabulary that every entry's keywords are drawn from. |

With only these, the **Wiki** (without scan images) and **Learning** pages work. Search shows "no data" and Q&A shows a random entry.

### 2. What to ask the team for

Pick the pages you want, then copy those files to **exactly** these paths:

| You want… | Copy these files | Put them in |
|---|---|---|
| **Search** | `embeddings.pkl`, `keywords.pkl` | `data/indexes/` |
| **Q&A** | `advice_index.pkl` and `advice.csv` (plus `CA_DAO_VAN_DAP_KEY` in `.env`) | `data/indexes/` and `data/advice/` |
| **Scan images on the Wiki** | the ~8,700 `<id>.png` images | `data/images/` |

### 3. Check it worked

Start the server (`make dev`) and read the startup log. It prints a line for each file it loads, such as `Loaded 687 advice entries`, and a `WARNING` line for each file it can't find. Fix the path of anything it warns about and restart.

You can also run `python -m tools.check_indexes` to sanity-check the search index files.

That's all a user of the site needs. The next section is only for people who want to regenerate the data.

---

## Repo layout

```
app/                       ★ the web app (Flask)
  server.py                  everything the site does
  i18n.py                    UI strings (vi / en)
  templates/  static/        frontend
pipeline/                  ★ steps that build the data (see below)
  split_spreads.py           Step 1: split 2-page spreads
  ocr_editor/                Step 2: OCR editor web tool (macOS)
  review_app/                Step 3: human review web tool
  enrich_metadata.py         Step 4: LLM explanation + keywords
  build_search_index.py      Step 5: search index
  build_advice_index.py      Step 6: Q&A index
tools/                     one-off helpers
  cut_pages.py               remove pages from a PDF
  apple_ocr.py               standalone Apple Vision OCR CLI (not part of the pipeline)
  check_indexes.py           sanity-check the search index files
  learning_data_csv.py       learning data JSON <-> CSV
  learning_data_text.py      learning data JSON <-> plain text
paths.py                   every data path, resolved from the repo root
data/
  source/                    scanned PDFs: 1.pdf, 2.pdf, 1-split.pdf, 2-split.pdf
  ocr/                       entries-extracted.csv (OCR editor output)
  images/                    <id>.png, the cropped scan of each entry
  review/                    input.csv, reviewed.csv, review_state.json
  enrichment/                keywords.json (in git), .keywords_embeddings.json (cache)
  wiki/                      wiki.csv (master dataset, in git), wiki.db (built from wiki.csv)
  indexes/                   embeddings.pkl, keywords.pkl, advice_index.pkl
  advice/                    advice.csv
  learning/                  learning-data.json, user_interpretations.txt (in git)
docs/
  guidelines/                rules for AI coding assistants
  agents/                    config for agent skills
GLOSSARY.md                domain terms
Makefile                   dev / prod / stop / status / logs
```

---

## Rebuilding the data from scratch (most people can skip this)

You only need this section if you are changing the dataset itself: fixing OCR errors, adding entries, changing the explanations, or re-tuning the search index. Everything in it has already been run once and its results are the files above.

**Usually you only need Step 5 and/or Step 6**, after editing `wiki.csv` or `advice.csv`. Steps 1–4 mean redoing OCR and manual review of thousands of entries, and Step 4 costs OpenAI money.

### Files the pipeline uses

These are also in shared storage. Ask for the ones the step you are running reads.

| File | Size / rows | Made by | Read by |
|---|---|---|---|
| `data/source/1.pdf`, `2.pdf` | 2 volumes | the original scanned dictionary | Step 1 |
| `data/source/1-split.pdf`, `2-split.pdf` | | Step 1 | Step 2 |
| `data/ocr/entries-extracted.csv` | | Step 2 | merged by hand into `data/review/input.csv` |
| `data/review/input.csv` | | merged by hand | Step 3 |
| `data/review/reviewed.csv` | | Step 3 | Step 4 |
| `data/enrichment/keywords.json` | ~76 keywords | Step 4 (grows as it runs) | Step 4 |
| `data/wiki/wiki.csv` | 8,651 rows | Step 4 | Step 5; rebuilding `wiki.db` |
| `data/advice/advice.csv` | 687 rows | curated subset of `wiki.csv` (how it was made is not recorded) | Step 6 |

### Main CSV schemas

- `data/wiki/wiki.csv` is the master dataset: `id, Alphabet, poem, category, explanation, pages, searchable_explanation, keywords`
  - `id` is a UUID. It is also the image name in `data/images/`.
  - `category` is one of `(CA DAO)`, `(TỤC NGỮ)`, `(THÀNH NGỮ)`, and so on.
  - `explanation` is the original dictionary text.
  - `searchable_explanation` is a 2–3 sentence LLM summary, used for embedding.
  - `keywords` is a comma-separated list drawn from `data/enrichment/keywords.json`.
- `data/advice/advice.csv`: `id, poem, category, searchable_explanation, keywords`
- `data/review/input.csv` → `data/review/reviewed.csv`: `id, poem, category, explanation, Quyển[, keyword, type]`

### The pipeline

```
 data/source/1.pdf, 2.pdf  (scanned dictionary, 2 volumes)
     │  Step 1: pipeline.split_spreads, tools.cut_pages      split 2-page spreads, drop junk pages
     ▼
 data/source/1-split.pdf, 2-split.pdf
     │  Step 2: pipeline.ocr_editor.app                      OCR editor (Apple Vision, macOS only)
     ▼
 data/ocr/entries-extracted.csv  +  data/images/<id>.png
     │  (both volumes merged by hand, "Quyển" column added)
     ▼
 data/review/input.csv
     │  Step 3: pipeline.review_app.app                      human review / cleanup UI
     ▼
 data/review/reviewed.csv
     │  Step 4: pipeline.enrich_metadata                     OpenAI adds searchable_explanation + keywords
     ▼
 data/wiki/wiki.csv  ──────────────────────────────►  data/wiki/wiki.db (auto-built by the server)
     │  Step 5: pipeline.build_search_index
     ▼
 data/indexes/embeddings.pkl, keywords.pkl            ──►  Search

 data/advice/advice.csv
     │  Step 6: pipeline.build_advice_index
     ▼
 data/indexes/advice_index.pkl                        ──►  Q&A
```

### Step 1: Prepare the PDFs

```bash
pip install -r requirements-ocr.txt
python -m pipeline.split_spreads                            # data/source/2.pdf → 2-split.pdf
python -m tools.cut_pages data/source/<pdf> 1,2,51-53       # remove covers/blank pages (1-based); writes <name>_cut.pdf
```

`split_spreads.py` has its input (`2.pdf`) and output hard-coded in `main()`; edit the file names there to process volume 1.

### Step 2: OCR the scans into CSV (macOS only)

```bash
python -m pipeline.ocr_editor.app   # opens the OCR editor on http://127.0.0.1:5001
```

This is a web tool that uses Apple's Vision framework for Vietnamese OCR. You pick a PDF from `data/source/`, adjust the boxes around each entry, and export. Exporting writes `data/ocr/entries-extracted.csv` and crops each entry into `data/images/<id>.png`. Page edits are kept **in memory only**, so restarting the tool loses unsaved work.

### Step 3: Review and clean entries by hand

```bash
python -m pipeline.review_app.app   # http://localhost:4005
```

Reads `data/review/input.csv`, lets a reviewer step through and fix each entry, and writes `data/review/reviewed.csv`. Progress is saved in `data/review/review_state.json`, so you can stop and resume.

### Step 4: Add LLM metadata (searchable explanation + keywords)

Needs `CHATGPT_KEY` in `.env`. Uses `gpt-5-mini` for the text and `text-embedding-3-small` to merge near-duplicate keywords.

```bash
python -m pipeline.enrich_metadata --input data/review/reviewed.csv --output data/wiki/wiki.csv --workers 20
```

- `--input` and `--output` are required. There are no defaults, so a careless re-run can't regenerate (and pay for) the whole dataset.
- It is resumable: rows that already have both fields are skipped, and progress is saved periodically. To continue an interrupted run, point `--input` at the partial output file.
- Keywords are restricted to the list in `data/enrichment/keywords.json`. New keywords are only added when nothing in the list is ≥ 0.85 similar. Embeddings are cached in `data/enrichment/.keywords_embeddings.json`.
- This step costs OpenAI money (~8,600 rows).

### Step 5: Build the search index

```bash
python -m pipeline.build_search_index
python -m tools.check_indexes        # optional sanity check
```

Reads `data/wiki/wiki.csv` and writes `embeddings.pkl` and `keywords.pkl` to `data/indexes/`. Rows with no `searchable_explanation` or `keywords` are skipped. Each entry gets 3 vectors: entry text, meaning and keywords. Runs locally; there is no API cost.

After it finishes, restart the server.

### Step 6: Build the Q&A index

```bash
python -m pipeline.build_advice_index   # reads data/advice/advice.csv, writes data/indexes/advice_index.pkl
```

Restart the server afterwards.

### Rebuilding `data/wiki/wiki.db`

On startup the server imports `data/wiki/wiki.csv` into `data/wiki/wiki.db`, but only if the table is empty. `wiki.db` is not in git; commit changes to `wiki.csv` instead. If `wiki.csv` changes (because you edited it or pulled someone else's edit), delete `wiki.db` and restart the server to rebuild it. Otherwise the server keeps serving the old data.

---

## Learning data

The Learning page reads a JSON list of `{topic, poems: [{poem_text, blanks: [{word, options}], introduction, interpretation}]}` from `data/learning/learning-data.json`.

Two ways to edit it:

```bash
# JSON <-> spreadsheet-friendly CSV
python -m tools.learning_data_csv data/learning/learning-data.json data/learning/learning-data.csv            # export
python -m tools.learning_data_csv data/learning/learning-data.csv data/learning/learning-data.json --reverse  # import

# JSON <-> plain text (data/learning/learning-data.txt)
python -m tools.learning_data_text --out    # JSON -> text
python -m tools.learning_data_text --in     # text -> JSON
```

The CSV and text files are editing intermediates and are gitignored; the JSON is the source of truth.

User likes and dislikes are appended to `data/learning/user_interpretations.txt` as `POEM_ID | USERNAME | STATUS | TEXT`. This file is tracked in git and holds real user input, so be careful about what you commit.

---

## Environment variables (`.env`)

| Variable | Used by | Purpose |
|---|---|---|
| `CA_DAO_VAN_DAP_KEY` | `app/server.py` | OpenAI key for the Q&A page (model `gpt-5.1`). Without it, Q&A returns the top match's stock explanation and no generated advice. |
| `CHATGPT_KEY` | `pipeline/enrich_metadata.py` | OpenAI key for Step 4 |

`.env.example` only lists `CHATGPT_KEY`; add `CA_DAO_VAN_DAP_KEY` yourself.

---

## How search works (`app/server.py`)

1. Embed the query and search the multi-vector FAISS index (cosine ≥ 0.6).
2. Find the closest keywords (≥ 0.6) and add every entry tagged with them.
3. If there are still no results, relax the threshold to 0.25, then force-accept the top 3 keywords.
4. As a last resort, do a plain substring search.
5. Sort by score, apply the category filter, and look up each entry's `id` in `wiki.db`.

Q&A works like this: embed the concern, take the max score per entry over its 3 vectors, keep the top 20, rerank them with the CrossEncoder, keep the top 3, then let OpenAI pick one and write the advice as JSON.

The thresholds are constants at the top of `app/server.py`.

---

## Gotchas

1. **The Makefile has two virtualenvs.** `make dev` and `make prod` use `./venv`, while `make dev-local` uses `./.venv`.
2. **`/learning-page` and `/wiki-page` are not in the nav bar** (only Search and Q&A are). This looks intentional after the redesign in commit `38accc0`.

---

## Notes for AI coding assistants

See [`docs/guidelines/ai-assistants.md`](docs/guidelines/ai-assistants.md). In short: most of `data/` may not exist on the machine you are editing, so don't search for it or create mock data. Don't start the server or test the website unless you are explicitly asked to. Never read `.env`.
