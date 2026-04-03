import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(metrics_data, out_dir):
    cm = metrics_data.get('confusion_matrix', {})
    if not cm: return
    
    # Format: [[tn, fp], [fn, tp]]
    matrix = [[cm.get('tn', 0), cm.get('fp', 0)],
              [cm.get('fn', 0), cm.get('tp', 0)]]
              
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual (Used)')
    plt.xlabel('Predicted (Prefetched)')
    plt.title('Prefetch Prediction Confusion Matrix')
    
    out_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_classification(metrics_data, out_dir):
    class_metrics = metrics_data.get('classification_metrics', {})
    if not class_metrics: return
    
    # Plot standard classification metrics
    keys = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    values = [class_metrics.get(k, 0) for k in keys]
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=keys, y=values, palette='viridis')
    plt.ylim(0, 1.0)
    plt.title('LSTM Model Classification Metrics')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    out_path = os.path.join(out_dir, 'classification_metrics.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_graph.py <metrics_json_file>")
        sys.exit(1)
        
    metrics_file = sys.argv[1]
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        out_dir = os.path.dirname(metrics_file)
        plot_confusion_matrix(data, out_dir)
        plot_classification(data, out_dir)
