# evaluate_fusion_methods.py
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# test_texts: List[str]
# test_labels: List[Dict] e.g., [{"confidence": 0.8, "anxiety": 0.3, ...}, ...]

def evaluate_method(predict_func, test_texts, test_labels, method_name):
    """Evaluate a single method"""
    all_emotions = ["confidence", "satisfaction", "hope", "trust_medical", "anxiety", "anger"]
    
    y_true = {e: [] for e in all_emotions}
    y_pred = {e: [] for e in all_emotions}
    
    for text, label in zip(test_texts, test_labels):
        pred = predict_func(text)
        for emotion in all_emotions:
            y_true[emotion].append(1 if label.get(emotion, 0) >= 0.5 else 0)
            y_pred[emotion].append(1 if pred.get(emotion, 0) >= 0.5 else 0)
    
    f1_scores = {}
    for emotion in all_emotions:
        f1_scores[emotion] = f1_score(y_true[emotion], y_pred[emotion], zero_division=0)
    
    macro_f1 = np.mean(list(f1_scores.values()))
    
    # Calculate overall precision and recall
    y_true_flat = [v for e in all_emotions for v in y_true[e]]
    y_pred_flat = [v for e in all_emotions for v in y_pred[e]]
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    
    print(f"\n{method_name}:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Per-emotion F1: {f1_scores}")
    
    return macro_f1, precision, recall, f1_scores

# Run evaluation
# 1. Static Weighted
evaluate_method(ensemble_static, test_texts, test_labels, "Static Weighted")

# 2. Stacked Ensemble
stacked = StackedEnsemble()
stacked.train(val_texts, val_labels) 
evaluate_method(stacked.predict, test_texts, test_labels, "Stacked Ensemble")

# 3. Uncertainty-based
evaluate_method(ensemble_uncertainty_weighted, test_texts, test_labels, "Uncertainty-Based")