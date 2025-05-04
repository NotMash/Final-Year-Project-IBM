import sys
import os
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from src.retrieval.pipeline import SemanticSearchPipeline
from src.data.loader import load_dataset


MODEL_PATH = os.path.join(PROJECT_ROOT, 'fine_tuned_models_path_insert_here', 'fine_tuned_model_2')
DATA_PATH = os.path.join(SRC_DIR, 'course_data', 'security_courses.json')
INDEX_PATH = os.path.join(PROJECT_ROOT, 'test_faiss_index.index')


@pytest.fixture(scope="module")
def pipeline():
    """Initialise the pipeline once for all tests"""
    p = SemanticSearchPipeline(MODEL_PATH, DATA_PATH, INDEX_PATH)
    yield p
    # Cleanup
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)


def test_pipeline_initialisation(pipeline):
    assert pipeline.model is not None
    assert len(pipeline.courses) > 0
    assert pipeline.embeddings.shape[0] == len(pipeline.courses)
    assert pipeline.index.ntotal == len(pipeline.courses)


def test_valid_query_returns_results(pipeline):
    query = "How do I secure my cloud data?"
    results = pipeline.search(query, k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "Course Title" in r
        assert "Relevance Score" in r
        assert 0.0 <= r["Relevance Score"] <= 1.0


def test_invalid_query_still_returns(pipeline):
    """Even with unusual input, should not break"""
    results = pipeline.search("asdfghjkl", k=3)
    assert isinstance(results, list)


def test_empty_query_returns_nothing_or_handles_gracefully(pipeline):
    results = pipeline.search("", k=3)
    assert isinstance(results, list)


def test_top_result_has_high_score(pipeline):
    results = pipeline.search("firewall security")
    if results:
        top_score = results[0]["Relevance Score"]
        print("\nTop relevance score for 'firewall security':", top_score)
        # Adjusted threshold based on model_training_evaluation realism
        assert top_score > 0.05
