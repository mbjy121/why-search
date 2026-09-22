# Dual Search — arXiv Papers + GitHub Repos, with "Why It Matched"

A small search engine that queries **research paper abstracts (arXiv)**
and **open-source repository READMEs (GitHub)** at the same time, and for
every result, tells you *which words in your query actually caused the
match* — not just a similarity score.

Wrapped in a Streamlit web app with a "Papers / Repos / Both" filter.

<p align="center">
  <em>Type a query. See the papers and the code side by side.
  See exactly why each hit was picked.</em>
</p>

---

## Why I built it

I'm exploring **information retrieval** — the field behind every search
box you've ever typed into — and I wanted a hands-on project that goes
beyond a tutorial notebook.

The specific angle came from reading work by **Prof. Jie Zou at the
University of Electronic Science and Technology of China (UESTC)**, whose
research on conversational and interactive information retrieval keeps
returning to the same idea: a good search system should *explain itself*,
not just rank documents. So instead of building yet another "top-K
similarity" demo, this one surfaces the overlapping words behind every
hit — the smallest possible step toward an explainable retriever.

Bringing arXiv papers and GitHub repos into the same index is deliberate:
in real IR work you almost always want the paper **and** the code, and
today they live in two different places.

---

## What it does

- Pulls **420 paper abstracts** from arXiv across the topics
  *information retrieval*, *recommender systems*, and *natural language
  processing*.
- Pulls **412 public GitHub repositories** tagged
  `recommender-system`, `information-retrieval`, or `nlp`, along with
  their README text.
- Merges both into a single searchable dataset (**832 documents**).
- Builds a **TF-IDF** index (unigrams + bigrams, English stopwords
  removed) using scikit-learn.
- Ranks documents against a query using **cosine similarity**.
- For each result, computes the per-term contribution to that similarity
  and returns the specific overlapping words — the *"why it matched"*.
- Serves it all through a Streamlit UI with a Papers / Repos / Both
  filter, source badges, and inline highlighting of the matched words.

---

## How to run it

### 1. Get the code

```bash
git clone <this repo>
cd dual-search
```

### 2. Install dependencies

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

### 3. Build the dataset (only needed once)

Each step below writes to `data/` and takes a minute or two.

```bash
python fetch_arxiv.py     # -> data/papers.json    (~420 papers)
python fetch_github.py    # -> data/repos.json     (~412 repos)
python build_dataset.py   # -> data/combined.json  (~832 entries)
```

> **GitHub rate limits.** `fetch_github.py` works unauthenticated, but if
> you hit a `403`, set a personal access token first:
> `export GITHUB_TOKEN=ghp_...` (`$env:GITHUB_TOKEN="ghp_..."` on
> PowerShell).

### 4. Try the search from the terminal

```bash
python search.py "conversational search"
```

### 5. Launch the web app

```bash
python -m streamlit run app.py
```

Streamlit opens the UI at <http://localhost:8501>. Type a query, flip the
filter, click through to arXiv or GitHub.

---

## Project layout

```
dual-search/
├── fetch_arxiv.py      # pulls paper abstracts from the arXiv API
├── fetch_github.py     # pulls repo metadata + README raw text
├── build_dataset.py    # merges both into data/combined.json
├── search.py           # TF-IDF index + "why it matched" logic
├── app.py              # Streamlit UI
├── requirements.txt
├── README.md           # you are here
└── data/
    ├── papers.json
    ├── repos.json
    └── combined.json
```

---

## Example queries and real output

The three queries below are real runs against the finished system. Snippets
are shortened for the README — the app shows more context.

### 1. `vector database embeddings`

| # | Type | Title | Score | Matched words |
|---|------|-------|-------|----------------|
| 1 | REPO | **weaviate/weaviate** | 0.138 | *vector database, vector, database, embeddings* |
| 2 | REPO | **qdrant/qdrant** | 0.114 | *vector database, vector, database, embeddings* |
| 3 | REPO | **StarlightSearch/EmbedAnything** | 0.108 | *vector database, vector, database, embeddings* |
| 4 | REPO | **kreeben/resin** | 0.098 | *vector database, vector, database* |
| 5 | REPO | **treygrainger/ai-powered-search** | 0.095 | *vector database, vector, database, embeddings* |

> *"weaviate/weaviate Weaviate is an open-source **vector database** that
> stores both objects and vectors, allowing for the combination of
> **vector** search with structured filtering..."*

### 2. `conversational search chatbot`

| # | Type | Title | Score | Matched words |
|---|------|-------|-------|----------------|
| 1 | PAPER | *Towards Fair Conversational Recommender Systems* | 0.158 | *conversational* |
| 2 | PAPER | *Improving Ad-hoc Search Effectiveness for Conversational IR via Model Merging* | 0.147 | *conversational, search* |
| 3 | PAPER | *Adaptive Personalized Conversational Information Retrieval* | 0.133 | *conversational, search* |
| 4 | PAPER | *Explainable Information Retrieval in the Audit Domain* | 0.099 | *conversational, search* |
| 5 | PAPER | *VideolandGPT: A User Study on a Conversational Recommender System* | 0.094 | *conversational* |

> *"Improving Ad-hoc **Search** Effectiveness for **Conversational**
> Information Retrieval via Model Merging. Conversational information
> retrieval is challenging since it requires..."*

### 3. `diffusion models for recommendation`

| # | Type | Title | Score | Matched words |
|---|------|-------|-------|----------------|
| 1 | REPO | **nancheng58/Awesome-LLM4RS-Papers** | 0.109 | *models recommendation, recommendation, models* |
| 2 | REPO | **WLiK/LLM4Rec-Awesome-Papers** | 0.086 | *models recommendation, recommendation, models* |
| 3 | REPO | **westlake-repl/Recommendation-Systems-without-Explicit-ID-Features-A-Literature-Review** | 0.076 | *models recommendation, recommendation, models* |
| 4 | REPO | **meta-pytorch/torchrec** | 0.075 | *recommendation* |
| 5 | PAPER | *Post-Userist Recommender Systems: A Manifesto* | 0.066 | *recommendation* |

Notice how the top hits are correctly biased toward **models +
recommendation** even though the query used the exact phrase "diffusion
models for recommendation" — the index tells you honestly *what* it
matched on, which is the whole point of the "why" column.

---

## How the "why it matched" works, in one paragraph

Given a query and a document, both are turned into TF-IDF vectors over the
same vocabulary. Their cosine similarity is a dot product of L2-normalized
vectors. That dot product is a **sum of per-term contributions**, so if
you take the element-wise product of the two vectors, each nonzero entry
tells you exactly *how much a specific word contributed to the match*.
The app returns those terms, ranked (bigrams before unigrams), as the
`matched_words` list you see in each card.

---

## Notes and limitations

- Data is a **snapshot** at fetch time. Re-run the `fetch_*.py` scripts
  to refresh.
- TF-IDF only finds **lexical** overlap. "Conversational agent" won't
  match "chatbot" the way a semantic embedding model would — a natural
  next step is to add a dense-retrieval fallback.
- READMEs are trimmed to 8,000 chars to keep the combined file small.
- `fetch_github.py` reads README text from `raw.githubusercontent.com`
  to bypass the strict `/repos/{owner}/{repo}/readme` unauthenticated
  rate limit.

---

## Credits

Motivated by **Prof. Jie Zou (UESTC)** and the broader line of work on
conversational and explainable information retrieval. Data from
[arXiv](https://arxiv.org/) and [GitHub](https://github.com/), used
according to their public APIs.
