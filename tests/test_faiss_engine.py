import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import faiss
import tempfile
import pytest

from src.retrieval.faiss_engine import create_faiss_index, save_index


def test_create_faiss_index_small():
    """Test Flat index creation for small datasets (<100 vectors)."""
    embeddings = np.random.rand(10, 128).astype('float32')
    index = create_faiss_index(embeddings)

    # Accept IndexFlatIP (used for cosine similarity) or fallback
    assert isinstance(index, (faiss.IndexFlatIP, faiss.IndexFlatL2))
    assert index.ntotal == embeddings.shape[0]


def test_create_faiss_index_large():
    """Test IVF index creation for larger datasets (>=100 vectors)."""
    embeddings = np.random.rand(200, 128).astype('float32')
    index = create_faiss_index(embeddings)

    assert isinstance(index, faiss.IndexIVFFlat)
    assert index.is_trained
    assert index.ntotal == embeddings.shape[0]


def test_save_index_creates_file():
    """Test saving a FAISS index to disk and reading it back."""
    embeddings = np.random.rand(10, 128).astype('float32')
    index = create_faiss_index(embeddings)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    save_index(index, tmp_path)

    assert os.path.exists(tmp_path)

    loaded_index = faiss.read_index(tmp_path)
    assert loaded_index.ntotal == index.ntotal

    os.remove(tmp_path)
