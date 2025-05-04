import json
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import traceback


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()  # Convert to Python native type
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert array to list
        elif np.isnan(obj):
            return None  # Convert NaN to None
        return super(NumpyEncoder, self).default(obj)


def load_unseen_data(dataset_path):
    print("Loading unseen dataset...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "courses" in data:
        return data["courses"]
    else:
        raise ValueError("Dataset format not recognized.")


def extract_description(course):
    """Example fallback extraction: tries 'Embedded Activities' first, then fallback fields."""
    if "Embedded Activities" in course and course["Embedded Activities"]:
        parts = []
        for activity in course["Embedded Activities"]:
            content = activity.get("Content", [])
            for line in content:
                if line.strip():
                    parts.append(line.strip())
        if parts:
            return " ".join(parts)
    # Fallback incase
    duration = course.get("Duration", "").strip()
    learners = course.get("Learners Completed", "").strip()
    available = course.get("Available Since", "").strip()
    return f"Duration: {duration}. Learners: {learners}. Available: {available}."


def evaluate_model(model_name, model, courses):
    """
    Evaluate each course by creating a query like: "What is <Course Title>?"
    Then encode description & query separately, measure time + similarity.
    """
    print(f"\nEvaluating {model_name} on unseen data...")
    results = []
    total_time = 0
    num_embeddings = 0

    for course in courses:
        title = course.get("Course Title", "N/A")
        description = extract_description(course)
        query = f"What is {title}?"

        # heuristic based on description length
        desc_len = len(description)
        if desc_len > 0:
            expected_relevance = min(1.0, 0.5 + (desc_len / 1000))
        else:
            expected_relevance = 0.5

        # Time the embeddings
        start_time = time.time()
        course_embedding = model.encode(description, convert_to_numpy=True)
        query_embedding = model.encode(query, convert_to_numpy=True)
        end_time = time.time()

        encoding_time = end_time - start_time
        total_time += encoding_time
        num_embeddings += 2

        cosine_similarity = np.dot(course_embedding, query_embedding) / (
                np.linalg.norm(course_embedding) * np.linalg.norm(query_embedding)
        )
        normalized_similarity = (cosine_similarity + 1) / 2

        # Print immediate results
        print(f"  Course: {title}")
        print(f"    Description length: {desc_len}")
        print(f"    Expected relevance: {expected_relevance:.4f}")
        print(f"    Actual similarity: {normalized_similarity:.4f}")
        print(f"    Encoding time: {encoding_time:.4f}s")

        results.append({
            "model_training_evaluation": model_name,
            "title": title,
            "query": query,
            "expected_relevance": expected_relevance,
            "cosine_similarity": normalized_similarity,
            "description_length": desc_len
        })

    avg_embedding_time = total_time / num_embeddings if num_embeddings else 0
    return results, avg_embedding_time


def calculate_metrics(results):
    """Compute MSE, MAE, and Spearman correlation from the final results list."""
    expected = [res["expected_relevance"] for res in results]
    predicted = [res["cosine_similarity"] for res in results]

    mse = mean_squared_error(expected, predicted)
    mae = mean_absolute_error(expected, predicted)

    if len(set(expected)) <= 1 or len(set(predicted)) <= 1:
        spearman_corr = float('nan')
        print("Warning: constant input detected - correlation is undefined.")
    else:
        spearman_corr, p_value = spearmanr(expected, predicted)
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {p_value:.4f})")

    return mse, mae, spearman_corr


def visualize_results(results, model_name, output_dir):
    """
    Plot scatter of expected vs. actual similarity + a text box with stats.
    Plot stored in <model_name>_similarity.png
    """


    plt.figure(figsize=(10, 6))
    expected = np.array([r["expected_relevance"] for r in results])
    predicted = np.array([r["cosine_similarity"] for r in results])
    titles = [r["title"] for r in results]

    # Scatter plotts
    plt.scatter(expected, predicted, alpha=0.7, s=80)


    for i, title in enumerate(titles):
        if predicted[i] < 0.5:
            plt.annotate(title, (expected[i], predicted[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)

    # Diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Ideal Alignment")

    plt.xlabel("Expected Relevance")
    plt.ylabel("Cosine Similarity (Normalized)")
    plt.title(f"Expected vs. Actual Similarity - {model_name}")


    mse_val = mean_squared_error(expected, predicted)
    mae_val = mean_absolute_error(expected, predicted)
    box_text = (
        f"MSE: {mse_val:.4f}\n"
        f"MAE: {mae_val:.4f}\n"
        f"Mean Similarity: {np.mean(predicted):.4f}\n"
        f"Min Similarity: {np.min(predicted):.4f}\n"
        f"Max Similarity: {np.max(predicted):.4f}"
    )
    plt.figtext(0.15, 0.15, box_text, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{model_name}_similarity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def visualize_model_comparison(models, metric_values, metric_name, color, output_dir, filename):
    """
    Bar chart comparing the given metric across fine_tuned_models_path_insert_here.
    The style and labeling are from your original code snippet.
    """


    plt.figure(figsize=(12, 6))


    plot_vals = [0 if np.isnan(x) else x for x in metric_values]

    bars = plt.bar(models, plot_vals, color=color, alpha=0.7)


    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(metric_values[i]):
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f"{metric_values[i]:.4f}", ha='center', va='bottom', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     "N/A", ha='center', va='bottom', fontsize=9)

    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by Model")
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def create_correlation_matrix(models, all_model_results, output_dir):
    """
    Create a Spearman correlation matrix between model_training_evaluation predictions.
    """

    # Gatherin model_training_evaluation predictions
    model_predictions = {m: [] for m in models}
    for res in all_model_results:
        model_predictions[res["model_training_evaluation"]].append(res["cosine_similarity"])

    corr_matrix = np.zeros((len(models), len(models)))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p1 = model_predictions[m1]
                p2 = model_predictions[m2]
                if len(p1) == len(p2) and len(p1) > 0:
                    try:
                        corr, _ = spearmanr(p1, p2)
                        if not np.isnan(corr):
                            corr_matrix[i, j] = corr
                    except:
                        corr_matrix[i, j] = 0

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Spearman Correlation')
    plt.title("Correlation Between Model Predictions")
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.yticks(range(len(models)), models)

    for i in range(len(models)):
        for j in range(len(models)):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}",
                     ha='center', va='center',
                     color='black' if abs(corr_matrix[i, j]) < 0.7 else 'white')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "model_correlation_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def create_performance_radar(models, metrics, output_dir):
    """
    Quick radar chart comparing model_training_evaluation performance across certain metrics.
    The normalization logic is the same from your original code snippet.
    """

    metrics_names = list(metrics.keys())
    num_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close circle

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for model_name in models:
        idx = models.index(model_name)
        vals = []
        for mkey in metrics_names:
            vals.append(metrics[mkey][idx])

        vals = [0 if np.isnan(x) else x for x in vals]
        # close the loop
        vals += vals[:1]

        # Basic invert for MSE, MAE, Time
        normalized = []
        for i, val in enumerate(vals):
            metric = metrics_names[i % len(metrics_names)]
            if metric in ["MSE", "MAE", "Embedding Time"]:
                # lower is better -> invert
                all_vals = [v for v in metrics[metric] if not np.isnan(v)]
                if all_vals:
                    mx = max(all_vals)
                    if mx > 0:
                        normalized.append(1 - (val / mx))
                    else:
                        normalized.append(0.5)
                else:
                    normalized.append(0)
            else:
                # higher is better
                all_vals = [v for v in metrics[metric] if not np.isnan(v)]
                if all_vals:
                    mx = max(all_vals)
                    if mx > 0:
                        normalized.append(val / mx)
                    else:
                        normalized.append(0.5)
                else:
                    normalized.append(0)

        ax.plot(angles, normalized, linewidth=2, label=model_name)
        ax.fill(angles, normalized, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticklabels([])
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Comparison")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "model_radar_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def print_summary_statistics(models, all_metrics):
    """
    Print summary table highlighting the best model_training_evaluation (with double-asterisks) for each metric.
    """
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS".center(80))
    print("=" * 80)


    metric_info = [
        ("Mean Similarity", all_metrics["mean_similarity"], False),
        ("Max Similarity", all_metrics["max_similarity"], False),
        ("Min Similarity", all_metrics["min_similarity"], False),  # or True if you prefer highest minimum
        ("MSE", all_metrics["mse"], True),
        ("MAE", all_metrics["mae"], True),
        ("Embedding Time (s)", all_metrics["embedding_time"], True),
        ("Spearman Correlation", all_metrics["sts_correlation"], False)
    ]


    best_idx_for_metric = {}
    for (metric_label, values, lower_is_better) in metric_info:
        arr = np.array(values)

        if lower_is_better:
            arr = np.where(np.isnan(arr), np.inf, arr)
            best_idx = arr.argmin()
        else:
            arr = np.where(np.isnan(arr), -np.inf, arr)
            best_idx = arr.argmax()
        best_idx_for_metric[metric_label] = best_idx

    # Table headerss
    col_headers = [m[0] for m in metric_info]
    header_line = f"{'Model':<25}" + "".join(f"{h:<20}" for h in col_headers)
    print(header_line)
    print("-" * len(header_line))

    # Print rowss
    for i, model_name in enumerate(models):
        row_str = f"{model_name:<25}"
        for (metric_label, values, _) in metric_info:
            val = values[i]
            if np.isnan(val):
                cell = "N/A"
            else:
                cell = f"{val:.4f}"
            # highlight if best
            if i == best_idx_for_metric[metric_label]:
                cell = f"**{cell}**"
            row_str += f"{cell:<20}"
        print(row_str)

    print("-" * len(header_line))
    print("NOTE: ** indicates best model_training_evaluation for each metric")
    print("=" * 80)


def main():
    # Models comparison
    model_paths = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "paraphrase-distilroberta-base-v1": "sentence-transformers/paraphrase-distilroberta-base-v1",
        "bert-base-uncased": "bert-base-uncased",
        "roberta-base": "roberta-base",
        "gtr-t5-base": "sentence-transformers/gtr-t5-base",

        # Please insert them in the correct folder
        "fine-tuned-model": "../../fine_tuned_models_path_insert_here/fine_tuned_model",
        "fine-tuned-model_2": "../../fine_tuned_models_path_insert_here/fine_tuned_model_2"


    }

    output_dir = "embedding_results"
    os.makedirs(output_dir, exist_ok=True)
    #load unnseen fata cfrom course data fodler

    dataset_path = "../course_data/security_courses.json"

    try:
        courses = load_unseen_data(dataset_path)
        print(f"Loaded {len(courses)} courses from dataset.")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        return

    #  aggregated results for final stats
    models_list = []
    mean_sims = []
    max_sims = []
    min_sims = []
    mses = []
    maes = []
    stss = []
    times = []

    all_model_results = []

    for model_name, model_path in model_paths.items():
        try:
            print(f"\nLoading model_training_evaluation: {model_name} from {model_path}")
            model = SentenceTransformer(model_path)
            results, avg_time = evaluate_model(model_name, model, courses)
            all_model_results.extend(results)

            # Summaries
            mse, mae, spearman_corr = calculate_metrics(results)
            mean_sim = np.mean([r["cosine_similarity"] for r in results])
            max_sim = np.max([r["cosine_similarity"] for r in results])
            min_sim = np.min([r["cosine_similarity"] for r in results])

            # Print short
            print(f"\nModel: {model_name}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Spearman Corr: {spearman_corr:.4f}")
            print(f"  Mean Similarity: {mean_sim:.4f}")
            print(f"  Min Similarity: {min_sim:.4f}")
            print(f"  Max Similarity: {max_sim:.4f}")
            print(f"  Embedding Time (avg): {avg_time:.4f}s")

            models_list.append(model_name)
            mses.append(mse)
            maes.append(mae)
            stss.append(spearman_corr)
            mean_sims.append(mean_sim)
            max_sims.append(max_sim)
            min_sims.append(min_sim)
            times.append(avg_time)


            visualize_results(results, model_name, output_dir)

        except Exception as e:
            print(f"Error evaluating model_training_evaluation {model_name}:")
            traceback.print_exc()
            continue

    # Summaries
    summary = {
        "fine_tuned_models_path_insert_here": models_list,
        "metrics": {
            "mean_similarity": mean_sims,
            "max_similarity": max_sims,
            "min_similarity": min_sims,
            "mse": mses,
            "mae": maes,
            "embedding_time": times,
            "sts_correlation": stss
        }
    }

    # Save JSON
    json_path = os.path.join(output_dir, "model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)


    if models_list:
        print_summary_statistics(models_list, summary["metrics"])


    if models_list:
        visualize_model_comparison(models_list, mean_sims, "Mean Similarity Score", "blue",
                                   output_dir, "mean_similarity.png")
        visualize_model_comparison(models_list, max_sims, "Max Similarity Score", "orange",
                                   output_dir, "max_similarity.png")
        visualize_model_comparison(models_list, min_sims, "Min Similarity Score", "red",
                                   output_dir, "min_similarity.png")
        visualize_model_comparison(models_list, times, "Embedding Generation Time (s)", "purple",
                                   output_dir, "embedding_time.png")
        visualize_model_comparison(models_list, mses, "Mean Squared Error (MSE)", "green",
                                   output_dir, "mse.png")
        visualize_model_comparison(models_list, maes, "Mean Absolute Error (MAE)", "brown",
                                   output_dir, "mae.png")
        visualize_model_comparison(models_list, stss, "Spearman Correlation", "teal",
                                   output_dir, "spearman_correlation.png")

        # Correlation matrix
        create_correlation_matrix(models_list, all_model_results, output_dir)

        # Radarr
        rad_metrics = {
            "Mean Similarity": mean_sims,
            "Max Similarity": max_sims,
            "MSE": mses,
            "MAE": maes,
            "Embedding Time": times,
            "Spearman Corr": stss
        }
        create_performance_radar(models_list, rad_metrics, output_dir)

        print(f"\nEvaluation complete. Results saved to '{output_dir}' directory.")
    else:
        print("No results to visualize. Possibly no fine_tuned_models_path_insert_here were evaluated successfully.")


if __name__ == "__main__":
    main()
