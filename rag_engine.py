import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    df = pd.read_csv("data/college_data.csv")
    df["text"] = df.apply(
        lambda x: f"{x['college']} in {x['city']} offers {x['branch']} for {x['category']} category "
                  f"with cutoff rank {x['cutoff_rank']} under {x['seat_type']} seat "
                  f"and fees {x['fees']} in year {x['year']}",
        axis=1
    )
    return df

def build_vector_db(df):
    embeddings = model.encode(df["text"].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_colleges(query, df, index, top_k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[indices[0]]
