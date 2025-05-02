# pipeline.py (Refactored to use improved faiss_engine.py)

import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from src.retrieval.faiss_engine import create_faiss_index, save_index, load_index, search_index
from src.data.loader import load_dataset


class SemanticSearchPipeline:
    def __init__(self, model_path, data_path, index_path):
        print("Initializing SemanticSearchPipeline...")
        self.model_path = model_path
        self.data_path = data_path
        self.index_path = index_path

        # Load courses
        print("Loading courses...")
        self.courses = load_dataset(self.data_path)
        print(f"Loaded {len(self.courses)} courses")

        # Load model_training_evaluation
        print("Loading model_training_evaluation...")
        self.model = SentenceTransformer(self.model_path)
        print("Model loaded successfully")

        # Generate or load index
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            self.index = load_index(self.index_path)
            print("FAISS index loaded from disk")
        else:
            print("Generating embeddings...")
            texts = [self.extract_description(c) for c in self.courses]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            print(f"Generated embeddings shape: {self.embeddings.shape}")

            print("Creating FAISS index...")
            self.index = create_faiss_index(self.embeddings)
            save_index(self.index, self.index_path)
            print(f"FAISS index saved to: {self.index_path}")

    def extract_description(self, course: Dict[str, Any]) -> str:
        parts = []
        title = course.get('Course Title', '').strip()
        parts.extend([title] * 3)  # Weight title more heavily

        content = []
        for activity in course.get('Embedded Activities', []):
            activity_content = [c.strip() for c in activity.get('Content', []) if c.strip()]
            content.extend([c for c in activity_content if len(c.split()) > 3 and 'click' not in c.lower()])

        if content:
            parts.extend(content[:3])  # Take first 3 meaningful content pieces

        return ' '.join(parts)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print(f"\nSearching for: {query}")
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search using improved FAISS functions
            similarities, indices = search_index(self.index, query_embedding, k)
            
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                course = self.courses[idx]
                content = []
                for activity in course.get('Embedded Activities', []):
                    content.extend([c.strip() for c in activity.get('Content', []) if c.strip() and len(c.split()) > 3])

                results.append({
                    "Course Title": course.get("Course Title", "N/A"),
                    "Description": content[0][:200] + "..." if content else "N/A",
                    "URL": course.get("URL", "N/A"),
                    "Relevance Score": float(similarity)  # Convert to float for JSON serialization
                })
            
            search_time = time.time() - start_time
            print(f"Search completed in {search_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def interactive(self):
        print("\n=== Interactive Search ===")
        print("Enter your search queries below. Type 'exit' to quit.")
        print("Tips:")
        print("- Be specific in your queries (e.g., 'network security fundamentals' instead of just 'security')")
        print("- Include key terms related to your interest")
        print("- Try different phrasings if results aren't relevant\n")

        while True:
            query = input("\nEnter your search query: ").strip()
            if query.lower() == 'exit':
                break

            results = self.search(query)
            print("\nSearch results (sorted by relevance):")
            for i, res in enumerate(results):
                print(f"\n{i + 1}. {res['Course Title']}")
                print(f"   Description: {res['Description']}")
                print(f"   URL: {res['URL']}")
                print(f"   Relevance Score: {res['Relevance Score']:.2%}")
