import argparse
import yaml
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from ultralytics import YOLO

def compute_prf_from_cm(cm: np.ndarray):
    """Computes precision, recall, and F1-score from a confusion matrix."""
    # rows = true, cols = pred
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    return precision, recall, f1

def plot_bar_chart(values, labels, title, ylabel, output_path):
    """Plots and saves a sorted bar chart for a given metric."""
    plt.figure(figsize=(12, 7))
    order = np.argsort(values)[::-1]
    sorted_values = np.array(values)[order]
    sorted_labels = [labels[i] for i in order]
    
    sns.barplot(x=sorted_labels, y=sorted_values, palette='viridis')
    
    for i, v in enumerate(sorted_values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {title} chart to {output_path}")
    plt.close()

def plot_confusion_matrix(matrix, class_names, output_path, normalize=False):
    """Plots and saves the confusion matrix as a heatmap."""
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        matrix = matrix.astype(int)
        fmt = 'd'
        title = 'Confusion Matrix (Raw Counts)'

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {'normalized' if normalize else 'raw'} confusion matrix to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model and generate metrics.")
    parser.add_argument('--weights', type=str, default='model/best.pt', help='Path to model weights.')
    parser.add_argument('--data', type=str, default='yolo_params.yaml', help='Path to the data YAML file.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on (e.g., test, val).')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions.')
    args = parser.parse_args()

    # Ensure data file path is correct
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"Error: Data YAML file not found at {data_path.resolve()}")
        return

    # Load the model
    model = YOLO(args.weights)

    # Run validation
    print(f"Running evaluation on the '{args.split}' split with confidence threshold {args.conf}...")
    metrics = model.val(data=str(data_path), split=args.split, conf=args.conf)

    # Extract and validate confusion matrix
    if hasattr(metrics, 'confusion_matrix'):
        cm = metrics.confusion_matrix.matrix
        class_names = metrics.names
        if cm.shape[0] != len(class_names):
            print(f"Adjusting CM shape from {cm.shape} to match {len(class_names)} classes.")
            cm = cm[:len(class_names), :len(class_names)]

            
            # --- Confusion Matrices ---
            raw_cm_path = Path('confusion_matrix_raw.png')
            plot_confusion_matrix(cm, class_names, raw_cm_path, normalize=False)
            
            norm_cm_path = Path('confusion_matrix_normalized.png')
            plot_confusion_matrix(cm, class_names, norm_cm_path, normalize=True)

            # --- Precision, Recall, F1 Score ---
            precision, recall, f1 = compute_prf_from_cm(cm)
            
            print("\n--- Per-Class Metrics ---")
            header = f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
            print(header)
            print("-" * len(header))
            
            metrics_data = []
            for i, name in enumerate(class_names):
                print(f"{name:<20} | {precision[i]:<10.3f} | {recall[i]:<10.3f} | {f1[i]:<10.3f}")
                metrics_data.append([name, precision[i], recall[i], f1[i]])

            # Save metrics to CSV
            csv_path = Path('per_class_metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score'])
                writer.writerows(metrics_data)
            print(f"\nSaved per-class metrics to {csv_path}")

            # --- Generate Bar Charts ---
            plot_bar_chart(precision, class_names, 'Per-Class Precision', 'Precision', Path('precision_chart.png'))
            plot_bar_chart(recall, class_names, 'Per-Class Recall', 'Recall', Path('recall_chart.png'))
            plot_bar_chart(f1, class_names, 'Per-Class F1-Score', 'F1-Score', Path('f1_score_chart.png'))

    else:
        print("Could not find confusion matrix in the metrics results.")

    # Print other metrics
    print("\n--- Evaluation Metrics ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    main()
