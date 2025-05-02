
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn
import torch
import random


def load_all_training_data(domain_dir, max_folds=3):
    print(f"Loading all training data from '{domain_dir}'...")
    training_examples = []
    for fold_num in range(1, max_folds + 1):
        fold_path = os.path.join(domain_dir, f"training_data_fold{fold_num}.json")
        with open(fold_path, 'r') as f:
            data = json.load(f)
        for course in data['courses']:
            description = course['description']
            for query in course['queries']:
                label = float(query['label']) / 1.0
                training_examples.append(InputExample(texts=[query['query'], description], label=label))
    print(f"Loaded {len(training_examples)} training examples from {domain_dir}.")
    return training_examples


def load_all_validation_data(domain_dir, max_folds=3):
    print(f"Loading all validation data from '{domain_dir}'...")
    validation_examples = []
    for fold_num in range(1, max_folds + 1):
        fold_path = os.path.join(domain_dir, f"validation_data_fold{fold_num}.json")
        with open(fold_path, 'r') as f:
            data = json.load(f)
        for example in data['examples']:
            validation_examples.append(InputExample(texts=example['texts'], label=example['label'] / 1.0))
    print(f"Loaded {len(validation_examples)} validation examples from {domain_dir}.")
    return validation_examples


def evaluate_model(model, val_examples):
    print("Evaluating model_training_evaluation...")
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="validation_eval")
    metrics = evaluator(model)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    return metrics


def enhanced_evaluation(model, val_examples):
    print("Computing enhanced evaluation metrics...")
    predictions = []
    ground_truths = []
    for example in val_examples:
        embeddings = model.encode(example.texts, convert_to_numpy=True)
        cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        predictions.append(cosine_similarity)
        ground_truths.append(example.label)
    mse = np.mean((np.array(ground_truths) - np.array(predictions)) ** 2)
    mae = np.mean(np.abs(np.array(ground_truths) - np.array(predictions)))
    spearman_corr = np.corrcoef(np.argsort(ground_truths), np.argsort(predictions))[0, 1]
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Spearman Correlation (Manual): {spearman_corr:.4f}")
    return mse, mae, spearman_corr


def build_custom_model(base_model_path):
    word_embedding_model = models.Transformer(base_model_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=128,
        activation_function=nn.GELU(),
        bias=True
    )
    dropout = models.Dropout(dropout=0.2)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model, dropout])


def continuous_fine_tune(base_model_path, domain_dirs, epochs_per_domain=8, batch_size=8, learning_rate=2e-5, estimated_total_steps=1000, max_folds=3):
    print("Starting continuous fine-tuning over all domains...")
    model = build_custom_model(base_model_path)
    continuous_model_dir = f"{base_model_path}_continuous_new"
    os.makedirs(continuous_model_dir, exist_ok=True)

    for domain_dir in domain_dirs:
        domain_name = os.path.basename(domain_dir)
        print(f"\n--- Fine-tuning on domain: {domain_name} ---")

        train_examples = load_all_training_data(domain_dir, max_folds=max_folds)
        val_examples = load_all_validation_data(domain_dir, max_folds=max_folds)

        # Convert to triplet format for ranking-aware learning
        triplets = []
        for anchor in train_examples:
            positive = random.choice(train_examples)
            negative = random.choice(train_examples)
            triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[1], negative.texts[1]]))

        train_dataloader = DataLoader(triplets, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name=f"{domain_name}_eval")

        optimizer_params = {'lr': learning_rate, 'eps': 1e-6, 'correct_bias': True, 'weight_decay': 0.01}

        print(f"Fine-tuning on {domain_name} for {epochs_per_domain} epochs...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs_per_domain,
            warmup_steps=100,
            evaluation_steps=None,
            optimizer_class=AdamW,
            optimizer_params=optimizer_params,
            scheduler='WarmupLinear',
            output_path=os.path.join(continuous_model_dir, f"{domain_name}_phase"),
            show_progress_bar=True,
            use_amp=False
        )

        print(f"Validation Metrics for {domain_name}:")
        evaluate_model(model, val_examples)
        enhanced_evaluation(model, val_examples)

        checkpoint_path = os.path.join(continuous_model_dir, f"checkpoint_{domain_name}")
        model.save(checkpoint_path)
        print(f"Checkpoint for domain {domain_name} saved to {checkpoint_path}")

    final_model_path = os.path.join(continuous_model_dir, "fine_tuned_model_2")
    model.save(final_model_path)
    print(f"\nFinal continuously fine-tuned model_training_evaluation saved to {final_model_path}")
    print("Continuous Fine-Tuning Complete!")


def main():
    base_model_path = "sentence-transformers/all-MiniLM-L6-v2_continuous/fine_tuned_model_2"
    base_folds_dir = "folds"
    domain_folders = ["storage_security", "compliance", "network_security"]
    domain_dirs = [os.path.join(base_folds_dir, d) for d in domain_folders]

    continuous_fine_tune(
        base_model_path=base_model_path,
        domain_dirs=domain_dirs,
        epochs_per_domain=8,
        batch_size=8,
        learning_rate=2e-5,
        estimated_total_steps=1000,
        max_folds=3
    )


if __name__ == "__main__":
    main()

