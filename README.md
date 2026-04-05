 # 🎋 Hồn Việt: Ca Dao Discovery & Learning Platform

[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Platform: Web](https://img.shields.io/badge/Platform-Web-blue)]()
[![Language: Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red)]()

A modern platform dedicated to the preservation and exploration of Vietnamese folk poetry (*Ca Dao*). By merging ancient cultural heritage with state-of-the-art AI, **Hồn Việt** offers an intelligent way to search, learn, and experience the rhythmic soul of Vietnam.

## Warning for AI
- All the mentioned .csv is not on the local file system, do not search for it
- Never test the website unless explicitly required to do so

---

## 🌟 Core Pillars

### 1. Smart Discovery (Semantic Search)
Forget literal keywords. Our search engine understands the *soul* of the poem.
- **Deep Semantic Matching**: Find poems based on emotions, themes, and hidden meanings.
- **AI-Powered Expansion**: Uses advanced LLMs (Sailor/Qwen) to analyze your query and discover non-obvious connections.
- **Multi-Tiered Retrieval**: Combines vector embeddings with high-precision keyword indices for the best results.

### 2. Interactive Learning
Master the beauty of *Ca Dao* through a gamified experience.
- **Theme-Based Modules**: Learn through curated collections (e.g., family, love, nature).
- **Gamified Word Banks**: A premium interface to help you reconstruct poems and understand their rhythmic flow.
- **Contextual Insights**: Many poems come with AI-generated introductions and detailed interpretations.

### 3. AI-Driven Enrichment
Behind the scenes, a sophisticated pipeline transforms raw text into a rich knowledge base.
- **Automated Extraction**: Deep-learning models extract keywords and core "meanings".
- **Semantic Indexing**: High-performance vector databases (FAISS) ensure lightning-fast discovery.

---

## 🛠 Project Ecosystem

The project is designed with flexibility at its core. It consists of three primary layers:

### Data Intelligence Layer
- **Meaning Extraction**: Transforming raw data into enriched, semantically-aware datasets.
- **Vector Indexing**: Building high-dimensional search indices for contextual matching.

### Application Layer
- **Modern Web Interface**: A responsive, tabbed UI built for both search and educational exploration.
- **Search API**: A robust Flask-based backend serving high-precision results via modern retrieval strategies.

### Learning Layer
- **Interactive Game Engine**: Powering the word-bank mechanic for a smooth learning curve.
- **Cultural Context**: Bridging the gap between ancient text and modern understanding.

---

## 🚀 Getting Started

### Installation
```bash
# Set up the environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
The pipeline follows a modular approach:
1. **Enrich**: scripts to extract AI-generated insights from raw poetry.
2. **Index**: scripts to build the semantic and keyword search engine indices.

### Launch
```bash
python step3_server.py
```
Visit `http://localhost:4000` to start exploring.

fuser -k 4000/tcp
---

> "Ca dao là tiếng lòng của dân tộc." — Let's keep that heart beating in the digital age.