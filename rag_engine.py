import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    df = pd.read_csv("data/college_data.csv")

    df["text"] = df.apply(
        lambda x: (
            f"{x['college']} in {x['city']} offers {x['branch']} "
            f"for {x['category']} category with cutoff rank {x['cutoff_rank']} "
            f"under {x['seat_type']} seat and fees {x['fees']} "
            f"in year {x['year']}"
        ),
        axis=1
    )
    return df

def build_vector_db(df):
    embeddings = model.encode(
        df["text"].tolist(),
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index   # ✅ RETURN ONLY INDEX

def retrieve_colleges(query, df, index, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return df.iloc[indices[0]]
