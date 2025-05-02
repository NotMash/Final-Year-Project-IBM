import os
import json
import matplotlib.pyplot as plt


def extract_main_scores(json_path):
    """
    Extract main scores from a single MTEB result JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Extract first test scores
            test_scores = data['scores']['test'][0]
            return {
                'accuracy': test_scores['accuracy'],
                'f1': test_scores['f1'],
                'f1_weighted': test_scores['f1_weighted']
            }
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None


def plot_model_metrics(results, output_dir='results/combined'):
    """
    Create bar charts for different metrics across fine_tuned_models_path_insert_here.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['accuracy', 'f1', 'f1_weighted']
    for metric in metrics:
        plt.figure(figsize=(14, 7))

        # Extract values for the current metric
        model_names = list(results.keys())
        metric_values = [results[model][metric] for model in model_names]

        # Create bar plot
        bars = plt.bar(model_names, metric_values, color='skyblue', edgecolor='navy')

        # Customize plot
        plt.title(f'{metric.upper()} Scores on Banking77Classification', fontsize=15)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)  # Scores typically range from 0 to 1

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_bar_chart.png'), dpi=300)
        plt.close()


def main():
    # Explicitly defined paths
    model_files = {
        "all-MiniLM-L6-v2": "results/all-MiniLM-L6-v2/sentence-transformers__all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/all-MiniLM-L6-v2.json",
        "all-mpnet-base-v2": "results/all-mpnet-base-v2/sentence-transformers__all-mpnet-base-v2/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0/all-mpnet-base-v2.json",
        "bert-base-uncased": "results/bert-base-uncased/google-bert__bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594/bert-base-uncased.json",
        "fine_tuned_model": "results/fine_tuned_sbert/no_model_name_available/no_revision_available/fine_tuned_sbert.json",
        "fine_tuned_model_2": "results/fine_tuned_sbert_2/no_model_name_available/no_revision_available/Banking77Classification.json",
        "gtr-t5-base": "results/gtr-t5-base/sentence-transformers__gtr-t5-base/9801579ce813fb37541e6098dffd17959fffcd6e/gtr-t5-base.json",
        "paraphrase-distilroberta-base-v1": "results/paraphrase-distilroberta-base-v1/sentence-transformers__paraphrase-distilroberta-base-v1/48bffbbd27bf028ecdd0cd55abb51236ec12ef1b/paraphrase-distilroberta-base-v1.json",
        "roberta-base": "results/roberta-base/FacebookAI__roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b/roberta-base.json"
    }

    # Extract scores for each model_training_evaluation
    model_results = {}
    for model_name, json_path in model_files.items():
        scores = extract_main_scores(json_path)
        if scores:
            model_results[model_name] = scores

    # Generate plots
    plot_model_metrics(model_results)

    print("✅ Bar charts generated successfully!")
    print("Models evaluated:", ", ".join(model_results.keys()))

    # Print out detailed scores for reference
    for model, metrics in model_results.items():
        print(f"\n{model} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()