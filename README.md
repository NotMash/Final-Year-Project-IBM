# Final Year Project – IBM SkillsBuild Semantic Search

This repository contains the complete source code for a final year Computer Science project focused on building a semantic search engine for IBM SkillsBuild’s cybersecurity courses. It uses SBERT embeddings, FAISS indexing, and a custom fine-tuning + evaluation pipeline benchmarked using MTEB.

---
##  Download The Models here
https://huggingface.co/notMash/fined_tuned_model/tree/main

## Note ^ This is not needed if you're cloning from hugging_face directly

##  Project Structure

```
Final-Year-Project-IBM-SkillsBuild-Final/
├── fine_tuned_models_path_insert_here/
│   ├── fine_tuned_model/
│   ├── fine_tuned_model_2/
│   └── .gitkeep
│
├── folds/
│   ├── compliance/
│   ├── network_security/
│   ├── original/
│   └── storage_security/
│
├── MTEB/
│   ├── results/
│   ├── mteb_test.py
│   └── mtebBarChart.py
│
├── src/
│   ├── configs/
│   │   └── config.yaml
│   │
│   ├── course_data/
│   │   ├── security_courses.json
│   │   └── unseen_data.json
│   │
│   ├── data/
│   │   └── loader.py
│   │
│   ├── model_training_evaluation/
│   │   ├── train.py
│   │   ├── evaluation.py
│   │   └── embedding_results/
│   │
│   └── retrieval/
│       ├── __init__.py
│       ├── faiss_engine.py
│       ├── pipeline.py
│       └── main.py
│
├── tests/
│   ├── test_faiss_engine.py
│   ├── test_loader.py
│   └── test_pipeline.py
│
├── faiss_index.index
├── .env
├── .gitignore
├── .gitattributes
├── environment.yml
├── environment_windows.yml
├── requirements.txt
└── README.md

```

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NotMash/Final-Year-Project-IBM.git
cd Final-Year-Project-IBM
```

### 2. Create Conda Environment Mac Or Windows

To set up your environment using the included YAML file:

### 2.1. MacOS
```bash on MacOS
conda env create -f environment.yml
conda activate finalyearproject
```
### 2.1. Windows
```bash on Windows
conda env create -f environment_windows.yml
conda activate finalyearproject
```

## Note Set The Conda As the interpreter

Alternatively, if you are using pip but I recommend CONDA:

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

##  Important Step

> Before running the pipeline, **you must delete the `.gitkeep` file** and add the fine-tuned SentenceTransformer models.

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
