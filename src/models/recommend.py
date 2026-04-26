import os
# 🔥 Prevent crashes on macOS (threading issues)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config

# -----------------------------
# LOAD CONFIG
# -----------------------------
config = load_config()

# -----------------------------
# LOAD DATA (safe)
# -----------------------------
df = pd.read_csv(config["data"]["processed"])
df = df.reset_index(drop=True)

# -----------------------------
# LOAD FAISS INDEX (safe)
# -----------------------------
faiss_path = "models/faiss_index.bin"
if not os.path.exists(faiss_path):
    raise FileNotFoundError("FAISS index not found. Run embeddings step first.")

index = faiss.read_index(faiss_path)

# -----------------------------
# LOAD MODEL ONCE (IMPORTANT)
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def generate_learning_path(user_input, min_similarity=0.4):

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------
    if not user_input:
        return pd.DataFrame(columns=["course_name"])

    user_input = [str(s).lower().strip() for s in user_input if s]

    if len(user_input) == 0:
        return pd.DataFrame(columns=["course_name"])

    # -----------------------------
    # EMBEDDING (safe)
    # -----------------------------
    user_embedding = model.encode(
        [" ".join(user_input)],
        convert_to_numpy=True
    ).astype("float32")

    # -----------------------------
    # 🔥 FAISS SEARCH
    # -----------------------------
    k = min(50, len(df))  # safety
    D, I = index.search(user_embedding, k)

    indices = I[0]
    distances = D[0]

    # -----------------------------
    # MAP RESULTS
    # -----------------------------
    df_temp = df.iloc[indices].copy()

    # Avoid division issues
    distances = np.maximum(distances, 1e-10)

    # Convert distance → similarity
    df_temp["similarity"] = 1 / (1 + distances)

    # -----------------------------
    # HYBRID SCORING
    # -----------------------------
    if "popularity" not in df_temp.columns:
        df_temp["popularity"] = 0

    df_temp["final_score"] = (
        0.7 * df_temp["similarity"] +
        0.3 * df_temp["popularity"]
    )

    # -----------------------------
    # FILTER + SORT
    # -----------------------------
    df_temp = df_temp[df_temp["similarity"] >= min_similarity]

    if df_temp.empty:
        return pd.DataFrame(columns=["course_name"])

    df_temp = df_temp.sort_values(
        "final_score",
        ascending=False
    ).head(10)

    return df_temp[
        ["course_name", "similarity", "popularity", "final_score"]
    ]