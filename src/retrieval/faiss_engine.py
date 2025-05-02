import faiss
import numpy as np
from sklearn.preprocessing import normalize

def create_faiss_index(embeddings, nlist=None, nprobe=5):
    """
    Create an optimized FAISS index with:
    - For small datasets (< 100 vectors): Uses IndexFlatIP for direct cosine similarity
    - For larger datasets: Uses IVF with quantization
    
    Args:
        embeddings: numpy array of embeddings
        nlist: number of clusters (if None, will be set based on data size)
        nprobe: number of clusters to visit during search
    """
    # Normalize embeddings for cosine similarity
    embeddings = normalize(embeddings, axis=1, norm='l2')
    
    # Get embedding dimension and number of vectors
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    
    # For small datasets, use simple flat index
    if num_vectors < 100:
        print(f"Using flat index for small dataset ({num_vectors} vectors)")
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        index.add(embeddings)
        return index
    
    # For larger datasets, use IVF
    if nlist is None:
        nlist = min(100, max(1, num_vectors // 4))
    
    try:
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF index with quantization
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        index.train(embeddings)
        
        # Add vectors to the index
        index.add(embeddings)
        
        # Set number of probes for search
        index.nprobe = min(nprobe, nlist)
        
        return index
    except Exception as e:
        print(f"Error creating IVF index: {str(e)}")
        print("Falling back to flat index")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

def save_index(index, path):
    """Save FAISS index to disk"""
    try:
        faiss.write_index(index, path)
    except Exception as e:
        print(f"Error saving index: {str(e)}")
        raise

def load_index(path):
    """Load FAISS index from disk"""
    try:
        return faiss.read_index(path)
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        raise

def search_index(index, query_embedding, k=5):
    """
    Search the index with normalized query embedding
    Returns distances and indices
    """
    try:
        # Normalize query embedding
        query_embedding = normalize(query_embedding.reshape(1, -1), axis=1, norm='l2')
        
        # Search the index
        distances, indices = index.search(query_embedding, k)
        
        # Convert distances to similarity scores (cosine similarity)
        similarities = (1 - distances) / 2
        
        return similarities, indices
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return np.zeros((1, k)), np.zeros((1, k), dtype=np.int64)

# tests/test_model.py
# Add this to src/model_training_evaluation/train.py

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import torch.nn as nn

class SBERTFineTuner:
    def __init__(self, base_model_path):
        self.model = self.build_model(base_model_path)

    def build_model(self, path):
        word_embedding_model = models.Transformer(path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        dense = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=128,
            activation_function=nn.GELU()
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense])



def test_model_builds():
    model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    tuner = SBERTFineTuner(model_path)
    assert tuner.model is not None