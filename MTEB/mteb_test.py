import mteb
from sentence_transformers import SentenceTransformer

#
models = {
    # "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    # "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    # "paraphrase-distilroberta-base-v1": "sentence-transformers/paraphrase-distilroberta-base-v1",
    # "bert-base-uncased": "bert-base-uncased",
    # "roberta-base": "roberta-base",
    # "gtr-t5-base": "sentence-transformers/gtr-t5-base"
    # "fine_tuned_sbert": "INSERT PATH",
    # "fine-tuned-model_2": "INSERT PATH"

}

# elected the tasks
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)

# run evaluation for each model_training_evaluation
for model_name, model_path in models.items():
    print(f"Evaluating model: {model_name}")
    model = SentenceTransformer(model_path)
    evaluation.run(model, output_folder=f"mteb_results/{model_name}")
