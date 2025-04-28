import numpy as np
import streamlit as st
import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer
import torch
import re

# File paths
FAISS_INDEX_FILE = "faiss_index.index"
EMBEDDINGS_FILE = "embeddings.npy"
DATA_FILE = "arxiv_metadata.csv"

# Initialize session state for tracking initialization and messages
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.index = None
    st.session_state.data = None
    st.session_state.corpus = None
    st.session_state.corpus_length = 0

if "messages_shown" not in st.session_state:
    st.session_state.messages_shown = {
        "device_info": False,
        "index_info": False
    }

# Set device to MPS (Apple Silicon GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if not st.session_state.messages_shown["device_info"]:
    st.sidebar.info(f"Using **{'GPU (MPS)' if device.type == 'mps' else 'CPU'}** for embedding generation.")
    st.session_state.messages_shown["device_info"] = True

# Load SentenceTransformer model globally
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2").to(device)

model = load_model()

# Load Dataset
@st.cache_data
def load_data():
    with st.spinner("Loading dataset..."):
        try:
            data = pd.read_csv(DATA_FILE, usecols=["id", "title", "abstract", "categories", "authors", "doi", "versions"])
            data["year"] = data["versions"].apply(lambda x: re.search(r"\b(\d{4})\b", str(x)).group(1) if re.search(r"\b(\d{4})\b", str(x)) else None)
            data["title"] = data["title"].fillna("Untitled")
            data["abstract"] = data["abstract"].fillna("")
            return data
        except FileNotFoundError:
            st.error(f"Dataset file '{DATA_FILE}' not found!")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()

# Cache Corpus Creation
@st.cache_data
def create_corpus(data):
    with st.spinner("Creating corpus..."):
        return [f"{title}. {abstract}" for title, abstract in zip(data["title"], data["abstract"])]

# Save FAISS Index
def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_FILE)

# Load FAISS Index
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_FILE):
        return faiss.read_index(FAISS_INDEX_FILE)
    return None

# Compute Embeddings in Batches
def compute_embeddings(texts, batch_size=1024):
    embeddings = []
    with st.spinner("Generating embeddings..."):
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings).astype("float32")

# Update or Create FAISS Index
@st.cache_resource
def update_faiss_index(data, corpus, corpus_length):
    with st.spinner("Loading FAISS index..."):
        existing_index = load_faiss_index()
        if existing_index is not None and existing_index.ntotal == corpus_length:
            if not st.session_state.messages_shown["index_info"]:
                st.info("Using existing FAISS index.")
                st.session_state.messages_shown["index_info"] = True
            return existing_index

        if existing_index is None or len(corpus) > existing_index.ntotal:
            if existing_index is None:
                st.info("No FAISS index found. Computing embeddings for all entries...")
                embeddings = compute_embeddings(corpus)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
            else:
                st.info(f"New entries found: {len(corpus) - existing_index.ntotal}. Updating embeddings...")
                new_texts = corpus[existing_index.ntotal:]
                new_embeddings = compute_embeddings(new_texts)
                existing_index.add(new_embeddings)
                index = existing_index

            save_faiss_index(index)
            np.save(EMBEDDINGS_FILE, embeddings if existing_index is None else None)
            st.success("FAISS index updated and saved!")
        return load_faiss_index()

# Recommend Papers
def recommend_papers(user_input, index, data, top_n=5, keyword=None, selected_categories=None, selected_year=None, selected_authors=None):
    user_embedding = model.encode([user_input], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(user_embedding, top_n * 2)
    candidates = data.iloc[indices[0]]

    if keyword:
        candidates = candidates[
            candidates["title"].str.contains(keyword, case=False, na=False) |
            candidates["abstract"].str.contains(keyword, case=False, na=False)
        ]
    if selected_categories:
        candidates = candidates[candidates["categories"].str.contains("|".join(selected_categories), case=False, na=False)]
    if selected_year:
        candidates = candidates[candidates["year"] == selected_year]
    if selected_authors:
        candidates = candidates[candidates["authors"].str.contains(selected_authors, case=False, na=False)]

    return candidates.head(top_n) if not candidates.empty else "No papers found matching the criteria."

# Streamlit App UI
st.title("ArXiv Paper Recommendation System")
st.markdown("""
    Welcome to the ArXiv Paper Recommendation System! Enter a research interest or paste an abstract
    to get relevant paper recommendations from the arXiv dataset.
""")

# Initialize app (load data and FAISS index) only once
if not st.session_state.initialized:
    # Load data
    data = load_data()
    if data.empty:
        st.stop()
    
    # Create corpus
    corpus = create_corpus(data)
    corpus_length = len(corpus)

    # Load or update FAISS index
    index = update_faiss_index(data, corpus, corpus_length)

    # Store in session state
    st.session_state.data = data
    st.session_state.corpus = corpus
    st.session_state.corpus_length = corpus_length
    st.session_state.index = index
    st.session_state.initialized = True

# Retrieve from session state
data = st.session_state.data
corpus = st.session_state.corpus
index = st.session_state.index

# Input Boxes
user_input = st.text_area("Enter your research interest or a paper abstract:", height=150)
keyword = st.text_input("Enter a keyword (optional):")

# Filters
st.sidebar.header("Filters")
unique_categories = sorted(set(cat for cats in data["categories"].str.split() for cat in cats if cat))
selected_categories = st.sidebar.multiselect("Select categories (optional):", unique_categories)
selected_year = st.sidebar.selectbox("Select publication year (optional):", [None] + sorted(data["year"].dropna().unique()))
selected_authors = st.sidebar.text_input("Enter author name (optional):")
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# Generate Recommendations
if st.button("Recommend"):
    if user_input.strip():
        with st.spinner("Finding recommendations..."):
            recommendations = recommend_papers(
                user_input, index, data, top_n=top_n, keyword=keyword,
                selected_categories=selected_categories, selected_year=selected_year,
                selected_authors=selected_authors
            )
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.markdown("### Recommended Papers:")
            for _, row in recommendations.iterrows():
                st.write(f"**Title**: {row['title']}")
                st.write(f"**Abstract**: {row['abstract']}")
                st.write(f"**Categories**: {row['categories']}")
                st.write(f"**Authors**: {row['authors']}")
                st.write(f"**Publication Year**: {row['year']}")
                st.write(f"**ArXiv ID**: [{row['id']}](https://arxiv.org/abs/{row['id']})")
                if pd.notna(row["doi"]):
                    st.write(f"**DOI**: [{row['doi']}](https://doi.org/{row['doi']})")
                st.write("---")
    else:
        st.warning("Please enter some text to get recommendations!")