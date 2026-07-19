# AI Search Intelligence System

An end-to-end semantic search and research assistant for ML/AI arXiv papers. The system combines dense vector search, automatic summarization, keyword extraction, and an LLM-powered agent that can decide which tool to call in response to natural-language questions.

## Overview

This project builds a pipeline that:

1. Loads a corpus of ML arXiv papers (title + abstract).
2. Embeds each paper into a dense vector space using a sentence-transformer model.
3. Indexes the embeddings with FAISS for fast semantic similarity search.
4. Summarizes and extracts keywords from retrieved papers.
5. Exposes the search pipeline as a FastAPI service.
6. Wraps the pipeline in LangChain tools and connects them to an LLM (via Groq) to create an agent that can search, summarize, extract keywords, and compare papers conversationally.

## Features

- **Semantic Paper Search** — Encode a natural-language query and retrieve the most relevant papers by cosine similarity (via FAISS `IndexFlatIP`).
- **Automatic Summarization** — Generate concise summaries of retrieved abstracts.
- **Keyword Extraction** — Pull out the most relevant keyphrases from a paper using KeyBERT.
- **Paper Comparison** — Compare two papers based on their closest-matching abstracts using the LLM.
- **REST API** — A FastAPI backend (`main.py`) exposing a `/search_and_summarize/` endpoint.
- **Agentic Interface** — LangChain tools (`search_and_summarize`, `extract_keywords`, `compare_papers`) bound to a Groq-hosted LLM (`llama-3.1-8b-instant`) so the assistant can decide which tool to invoke based on a user's request.

## Tech Stack

| Component | Library / Model |
|---|---|
| Dataset | `CShorten/ML-ArXiv-Papers` (via 🤗 `datasets`) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Index | `faiss-cpu` (`IndexFlatIP`, cosine similarity via L2-normalized vectors) |
| Summarization | `transformers` text-generation pipeline (GPT-2 based) |
| Keyword Extraction | `KeyBERT` |
| API Layer | `FastAPI` + `uvicorn` + `nest_asyncio` |
| LLM / Agent | `langchain`, `langchain-groq` (`llama-3.1-8b-instant`), `langchain-huggingface` |
| Data Handling | `pandas`, `numpy` |

## Project Structure / Pipeline

The notebook (`AI_Search_Intelligence_System_project.ipynb`) walks through the full pipeline in stages:

1. **Data Preparation**
   - Load the `ML-ArXiv-Papers` dataset and convert to a DataFrame.
   - Keep `title` and `abstract`, sample the first 15,000 rows.
   - Combine `title` + `abstract` into a single `paper_text` field and clean whitespace/newlines.
   - Persist the processed DataFrame to `df.pkl`.

2. **Embedding Generation**
   - Load `all-MiniLM-L6-v2` from `sentence-transformers`.
   - Encode all paper texts into 384-dimensional embeddings.
   - Cache embeddings to `paper_embeddings.npy` to avoid recomputation.

3. **Vector Indexing (FAISS)**
   - L2-normalize embeddings and build a `faiss.IndexFlatIP` index (inner product ≈ cosine similarity on normalized vectors).
   - Persist the index to `paper_faiss.index`.
   - Implement `search_paper(query, k)` to retrieve the top-k most similar papers for a query.

4. **Summarization**
   - Use a `transformers` text-generation pipeline (GPT-2) to produce summaries of retrieved abstracts (a workaround since a dedicated summarization pipeline wasn't available in this environment).
   - Combine search + summarization into `search_and_summarize(query, k)`.

5. **Keyword Extraction**
   - Use `KeyBERT` (built on the same sentence-transformer model) to extract top keyphrases from any given text, with configurable n-gram range and stop-word filtering.

6. **FastAPI Service (`main.py`)**
   - Loads the DataFrame, embedding model, FAISS index, and text generator at startup, with error handling for missing resources.
   - `GET /shard` — health-check endpoint.
   - `POST /search_and_summarize/` — accepts `{"query": str, "k": int}` and returns matched papers with similarity score, title, abstract snippet, and generated summary.
   - Run in a background thread inside the notebook using `nest_asyncio` + `uvicorn`, and tested with `requests`.

7. **Agentic Layer (LangChain + Groq)**
   - Define LangChain `@tool`-decorated functions: `search_and_summarize`, `extract_keywords`, and `compare_papers`.
   - Bind tools to a Groq-hosted LLM (`llama-3.1-8b-instant`) via `ChatGroq`.
   - Build a `create_agent(model, tools)` agent capable of routing a natural-language request (e.g. *"Find the top 3 research papers on Vision Transformer"* or *"Extract the top 5 keywords from ..."*) to the correct tool and returning a final, tool-grounded response.
   - Demonstrates the manual tool-calling loop as well (`bind_tools` → inspect `tool_calls` → invoke tool → wrap in `ToolMessage` → feed back to the LLM for a final answer).

8. **Batch Keyword Extraction**
   - Apply keyword extraction across a sample of 100 papers and store results in a new DataFrame column (`extracted_keywords`) for inspection/demo purposes.

## Requirements

```
datasets
pandas
numpy
sentence-transformers
faiss-cpu
transformers
keybert==0.8.5
fastapi
uvicorn
nest-asyncio
requests
langchain-core
langchain-community
langchain-huggingface
langchain-groq
```

> **Note:** A Groq API key is required for the agentic layer (retrieved via `google.colab.userdata.get("researchpaperassistantproject")` when run in Google Colab). Set this up as an environment variable or secret if running outside Colab.

## Setup & Usage (Step by Step)

### Step 1 — Install dependencies
```bash
pip install datasets pandas numpy sentence-transformers faiss-cpu transformers \
    keybert==0.8.5 fastapi uvicorn nest-asyncio requests \
    langchain-core langchain-community langchain-huggingface langchain-groq
```
> `sentence-transformers` can conflict with `torchcodec`. If you hit an import error after installing, uninstall it with `pip uninstall -y torchcodec` and restart your runtime/kernel before continuing.

### Step 2 — Load and prepare the dataset
- Load `CShorten/ML-ArXiv-Papers` with Hugging Face `datasets` and convert it to a `pandas` DataFrame.
- Keep only the `title` and `abstract` columns, and (optionally) limit to the first 15,000 rows for a faster demo run.
- Create a combined `paper_text` column (`title` + `abstract`), strip newlines/whitespace.
- Save the cleaned DataFrame: `df.to_pickle("df.pkl")`.

### Step 3 — Generate sentence embeddings
- Load the embedding model: `SentenceTransformer("all-MiniLM-L6-v2")`.
- Encode `paper_text` for the whole dataset in batches (`model.encode(..., batch_size=32, show_progress_bar=True)`).
- Save the result: `np.save("paper_embeddings.npy", embeddings)`.
- On future runs, the code checks for this file first and loads it instead of re-encoding everything.

### Step 4 — Build the FAISS vector index
- L2-normalize the embeddings (`faiss.normalize_L2`) so inner product search behaves like cosine similarity.
- Create `faiss.IndexFlatIP(384)` and add the normalized embeddings.
- Persist it: `faiss.write_index(index, "paper_faiss.index")`.
- Sanity-check with `index.ntotal` to confirm the expected number of vectors were indexed.

### Step 5 — Test semantic search
- Encode a test query, normalize it, and call `index.search(query_embedding, k)`.
- Use the returned indices to look up `df.iloc[idx]["title"]` / `["abstract"]` and confirm the results look relevant.

### Step 6 — Add summarization
- Initialize a `transformers` text-generation pipeline (`pipeline("text-generation", model="gpt2")`) as a summarization stand-in.
- Combine retrieval + summarization into a single `search_and_summarize(query, k)` helper.

### Step 7 — Add keyword extraction
- Initialize `KeyBERT(model)` reusing the same sentence-transformer model.
- Call `kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=5)` on any abstract or query result.

### Step 8 — Stand up the FastAPI service
- Re-save the DataFrame (`df.to_pickle("df.pkl")`) so `main.py` can load it independently.
- Write `main.py` (via the `%%writefile main.py` cell), which loads the DataFrame, embedding model, FAISS index, and text-generation pipeline at startup, and exposes:
  - `GET /shard` — health check
  - `POST /search_and_summarize/` — body `{"query": str, "k": int}`
- **Inside the notebook:** run it in a background thread with `nest_asyncio` + `uvicorn.run(app, host="0.0.0.0", port=8000)`, then `time.sleep(5)` to let it boot.
- **Standalone (recommended for real use):**
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

### Step 9 — Call the API
```bash
curl -X POST http://localhost:8000/search_and_summarize/ \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning for medical image analysis", "k": 3}'
```
Or from Python:
```python
import requests
response = requests.post(
    "http://localhost:8000/search_and_summarize/",
    json={"query": "deep learning in medical imaging", "k": 3}
)
print(response.json())
```

### Step 10 — Set up the LLM agent
- Get a Groq API key from the Groq console.
  - In Colab, store it as a secret (e.g. `researchpaperassistantproject`) and load with `userdata.get(...)`.
  - Outside Colab, set it as an environment variable and load with `os.environ["GROQ_API_KEY"]`.
- Initialize the LLM:
  ```python
  from langchain_groq import ChatGroq
  llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)
  ```

### Step 11 — Define and register tools
- Define `search_and_summarize`, `extract_keywords`, and `compare_papers` as LangChain `@tool` functions (each wraps the underlying FAISS/KeyBERT/LLM logic from Steps 5–7).
- Collect them: `tools = [search_and_summarize, extract_keywords, compare_papers]`.

### Step 12 — Build and run the agent
```python
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find the top 3 research papers on Vision Transformer."}
    ]
})
print(response["messages"][-1].content)
```
- The agent automatically decides which tool to call based on the user's request.
- For manual control over tool calling (useful for debugging), use the low-level pattern instead:
  ```python
  llm_with_tools = llm.bind_tools(tools)
  response = llm_with_tools.invoke(user_query)
  tool_call = response.tool_calls[0]
  # dispatch tool_call["name"] -> matching tool.invoke(tool_call["args"])
  # wrap result in a ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
  # feed [SystemMessage, HumanMessage, response, tool_message] back into llm.invoke(...) for the final answer
  ```

### Step 13 (optional) — Batch keyword extraction demo
- Run `get_keywords_for_text` across a sample of papers (e.g. `df.head(100)`) and store results in a new `extracted_keywords` column to spot-check quality across many papers at once.

### Output files produced by this pipeline
| File | Produced in | Purpose |
|---|---|---|
| `df.pkl` | Step 2 | Cleaned paper metadata, reloaded by `main.py` |
| `paper_embeddings.npy` | Step 3 | Cached sentence embeddings |
| `paper_faiss.index` | Step 4 | Persisted FAISS vector index |
| `main.py` | Step 8 | FastAPI service source file |

## Known Limitations / Notes

- Summarization currently uses a GPT-2 text-generation pipeline as a stand-in for a true summarization model (e.g. BART/T5), since the `summarization` pipeline task wasn't available in the original environment. Summary quality is limited compared to a dedicated seq2seq summarizer.
- The dataset is capped at 15,000 papers for demonstration purposes; the full corpus can be used by removing the `df.head(15000)` limit.
- Embeddings and FAISS index are cached to disk (`paper_embeddings.npy`, `paper_faiss.index`) so they only need to be generated once.
- The notebook contains exploratory/duplicate cells (multiple iterations of `search_paper` / `search_and_summarize`) reflecting the iterative development process.

## Possible Improvements

- Swap the GPT-2 workaround for a proper summarization model (e.g. `facebook/bart-large-cnn`).
- Add authentication and rate-limiting to the FastAPI service.
- Persist agent conversation history for multi-turn research sessions.
- Add a lightweight front-end (e.g. Streamlit or React) on top of the FastAPI backend.
- Expand the agent's toolset (e.g. citation graph lookup, PDF export of results).
