from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Load NLP resources ---

df = pd.DataFrame() # Initialize empty DataFrame
try:
    if os.path.exists("df.pkl"):
        df = pd.read_pickle("df.pkl")
        if "paper_text" not in df.columns:
            df["paper_text"] = df["title"] + df["abstract"]
            df["paper_text"] = df["paper_text"].str.replace("\\n", " ", regex=False)
            df["paper_text"] = df["paper_text"].str.strip()
except Exception as e:
    print(f"Error loading DataFrame: {e}")

model = None
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")

# Initialize the text-generation pipeline as a workaround for summarization
text_generator = None
try:
    text_generator = pipeline("text-generation", model="gpt2")
    print("Text generation pipeline initialized successfully in FastAPI!")
except Exception as e:
    print(f"Error loading text-generation pipeline: {e}")

index = None
try:
    if os.path.exists("paper_faiss.index"):
        index = faiss.read_index("paper_faiss.index")
except Exception as e:
    print(f"Error loading FAISS index: {e}")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: int = 5

@app.get("/shard")
async def root():
    return {"message": "Hello from FastAPI!"}

@app.post("/search_and_summarize/")
async def run_search_and_summarize(request: QueryRequest):
    if model is None or text_generator is None or index is None or df.empty:
        raise HTTPException(status_code=503, detail="NLP resources not loaded. Server is not ready.")

    results = []
    try:
        query_embedding = model.encode([request.query])
        faiss.normalize_L2(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding query: {e}")

    try:
        D, I = index.search(query_embedding, request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing FAISS search: {e}")

    for score, idx in zip(D[0], I[0]):
        try:
            if idx >= len(df):
                continue

            abstract_text = df.iloc[idx]['abstract']

            summary_text = "Summarizer not available."
            if text_generator:
                # Use text_generator for summarization
                generated_summary = text_generator(
                    abstract_text,
                    max_length=120,
                    min_length=40,
                    num_return_sequences=1,
                    truncation=True
                )
                summary_text = generated_summary[0]["generated_text"]

            results.append({
                "score": float(score),
                "title": df.iloc[idx]['title'],
                "abstract_snippet": abstract_text[:500] + "...",
                "summary_text": summary_text,
            })
        except IndexError:
            pass
        except Exception as e:
            print(f"Error processing result for index {idx}: {e}")
            continue

    return {"query": request.query, "results": results}
