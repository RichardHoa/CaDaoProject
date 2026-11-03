# Rules for AI coding assistants

## Data is stored externally

Most files under `data/` (CSVs, PDFs, `data/images/`, the `.pkl` files in `data/indexes/`) live in the team's shared storage, not in git. They are usually missing from the machine you are editing. Don't search for them, and don't create mock data to stand in for them.

## Don't run the app

- Never start the server or test the website (e.g. `make dev`, `python -m app.server`) unless explicitly asked to.
- This repo is the code machine. The ML libraries (transformers, faiss, torch) and the venv live on the compute node, so don't look for them here.

## Secrets

Never read the `.env` file.
