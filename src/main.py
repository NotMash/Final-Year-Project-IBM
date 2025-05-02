# Refactored Project Entry Point for CLI Use

import argparse
import yaml
from sentence_transformers import SentenceTransformer
from src.retrieval.pipeline import SemanticSearchPipeline
import os
# To run main, python -m src.main --config src/configs/config.yaml 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Test model_training_evaluation loading first
    print("\n Testing model_training_evaluation loading...")
    model_path = config['model_path']
    print(f"🔍 Using model_training_evaluation from config: '{model_path}'")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📁 Does path exist? {os.path.exists(model_path)}")

    try:
        model = SentenceTransformer(model_path)
        print(" Model loaded successfully!")

        # Try a simple encoding
        text = "This is a test."
        embedding = model.encode(text)
        print(f"🧪 Test encoding shape: {embedding.shape}")
        print(" Model test successful!")

    except Exception as e:
        print(f" Error loading model_training_evaluation: {str(e)}")
        raise

    print("\n Initializing pipeline...")
    pipeline = SemanticSearchPipeline(**config)
    pipeline.interactive()


if __name__ == "__main__":
    main()
