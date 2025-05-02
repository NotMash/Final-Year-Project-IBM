# Final Year Project – IBM SkillsBuild Semantic Search

This repository contains the complete source code for a final year Computer Science project focused on building a semantic search engine for IBM SkillsBuild’s cybersecurity courses. It uses SBERT embeddings, FAISS indexing, and a custom fine-tuning + evaluation pipeline benchmarked using MTEB.

---
##  Download The Models here
https://huggingface.co/notMash/fined_tuned_model/tree/main


##  Project Structure

```
final-year-project/
├── fine_tuned_models_path_insert_here/
│   ├── .gitkeep                  # Placeholder (delete this)
│   ├── fine_tuned_model/      # add from hugging face first fine-tuned model
│   └── fine_tuned_model_2/      # add second from hugging face fine-tuned model
├── folds/                       # Cross-validation splits
├── MTEB/                        # MTEB benchmark results
├── src/
│   ├── course_data/             # JSON datasets (security_courses.json, unseen_data.json)
│   ├── data/                    # Dataset loaders
│   ├── model_training_evaluation/ # Training and evaluation scripts
│   └── retrieval/               # SemanticSearchPipeline and indexing
├── tests/                       # Unit tests for core functionality
├── environment.yml              # Conda environment definition
├── requirements.txt             # pip dependencies (if not using Conda)
├── .env                         # Optional environment variables
├── .gitignore                   # Files to ignore in Git
├── faiss_index.index            # Example index (can be regenerated)
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NotMash/Final-Year-Project-IBM.git
cd Final-Year-Project-IBM
```

### 2. Create Conda Environment

To set up your environment using the included YAML file:

```bash
conda env create -f environment.yml
conda activate finalyearproject
```

Alternatively, if you are using pip:

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🚨 Important Step

> Before running the pipeline, **you must delete the `.gitkeep` file** and add your fine-tuned SentenceTransformer models.

```bash
rm fine_tuned_models_path_insert_here/.gitkeep
```

Then, place your folders like so:

```
fine_tuned_models_path_insert_here/
├── fine_tuned_model/
├── fine_tuned_model_2/
```

Each folder should include `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.

---

## Running Main.py

You can main.py using:

```bash
python -m src.main --config src/configs/config.yaml
```

---

## Running Tests

You can run all tests using:

```bash
pytest tests/
```

---

## Notes

- All paths are relative and designed to work across machines.
- Ensure `security_courses.json` is present in `src/course_data/`.
- The FAISS index file will be regenerated if not found.

---

## License

This project is provided for educational use only. Commercial use is not permitted.
