import os
# 🔥 Prevent threading crashes on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mlflow
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.config import load_config

# ✅ Load config
config = load_config()

# ✅ Clean MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("course-recommender")

with mlflow.start_run():

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(config["data"]["processed"])
    df = df.dropna(subset=["combined_text"])

    # 🔧 (optional for testing stability)
    # df = df.head(300)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # -----------------------------
    # CREATE EMBEDDINGS (SAFE MODE)
    # -----------------------------
    texts = df["combined_text"].astype(str).tolist()

    embeddings_list = []
    for text in texts:
        emb = model.encode(text, convert_to_numpy=True)
        embeddings_list.append(emb)

    embeddings = np.array(embeddings_list).astype("float32")

    # -----------------------------
    # SAVE EMBEDDINGS
    # -----------------------------
    emb_path = config["model"]["embeddings"]
    np.save(emb_path, embeddings)

    # -----------------------------
    # 🔥 CREATE FAISS INDEX
    # -----------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss_path = "models/faiss_index.bin"
    faiss.write_index(index, faiss_path)

    # -----------------------------
    # LOG PARAMETERS
    # -----------------------------
    mlflow.log_param("model", "MiniLM")
    mlflow.log_param("num_samples", len(df))
    mlflow.log_param("embedding_dim", dimension)

    # -----------------------------
    # METRIC 1: Avg Similarity
    # -----------------------------
    sample_size = min(100, len(embeddings))

    if sample_size > 1:
        sim_matrix = cosine_similarity(embeddings[:sample_size])
        avg_similarity = float(np.mean(sim_matrix))
    else:
        avg_similarity = 0.0

    mlflow.log_metric("avg_similarity", avg_similarity)

    # -----------------------------
    # METRIC 2: Avg Norm
    # -----------------------------
    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = float(np.mean(norms))

    mlflow.log_metric("avg_embedding_norm", avg_norm)

    # -----------------------------
    # SAVE ARTIFACTS
    # -----------------------------
    mlflow.log_artifact(emb_path)
    mlflow.log_artifact(faiss_path)

    print("✅ Embeddings + FAISS index + metrics logged!")