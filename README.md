# Final Year Project – IBM SkillsBuild Semantic Course Search

This project is a semantic search system developed to improve the discovery of cybersecurity-related courses from IBM SkillsBuild. It indexes course descriptions and allows users to search using natural language queries, returning the most relevant matches based on semantic similarity.

This project was developed as part of a final-year Computer Science degree at Queen Mary University of London.

## Project Structure

```
final-year-project/
├── src/
│   ├── data/                      # Dataset loaders
│   ├── model_training_evaluation/ # Training and evaluation scripts
│   ├── retrieval/                 # Search pipeline and indexing
│   └── course_data/              # JSON dataset of courses
├── tests/                        # Unit tests
├── environment.yml               # Conda environment
├── requirements.txt              # pip-based dependencies
└── README.md
```

## Features

- Embeds and indexes cybersecurity course data.
- Provides a search interface using semantic similarity.
- Tested pipeline with unit coverage.
- Designed for reproducibility across machines.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/NotMash/Final-Year-Project-IBM.git
   cd Final-Year-Project-IBM
   ```

2. Create the environment using Conda:

   ```bash
   conda env create -f environment.yml
   conda activate finalyearproject
   ```

3. Run the test suite:

   ```bash
   pytest tests/
   ```

## Evaluation Metrics

The system was evaluated using unseen courses and assessed based on:

- Mean and max cosine similarity
- Mean Absolute Error (MAE)
- Spearman Correlation
- Embedding latency

Evaluation results are available in the `embedding_results/` directory.

## License

This project is distributed for academic and educational purposes. Please refer to the LICENSE file if included.
