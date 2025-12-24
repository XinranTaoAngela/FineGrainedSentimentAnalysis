"""
Improved Ensemble Methods for Healthcare Sentiment Analysis
Includes three ensemble strategies:
1. Static Weighted (original method)
2. Stacked Ensemble (Meta-learner)
3. Uncertainty-based Weighting
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

from classify import classify_with_roberta, classify_with_llm
from rule_based.rule_based import HealthcareSentimentDimensions

# Instantiate the rule-based model
rule_based_model = HealthcareSentimentDimensions()

all_emotions = ["confidence", "satisfaction", "hope", "trust_medical", "anxiety", "anger"]

# ============================================
# Method 1: Static Weighted Ensemble (original method)
# ============================================
def ensemble_static(text, weights=None, few_shot=True):
    """Original static weighted ensemble"""
    if weights is None:
        weights = {"roberta": 0.5, "rule": 0.2, "llm": 0.3}
    
    roberta_pred = classify_with_roberta(text)
    rule_pred = rule_based_model.analyze_text(text)
    llm_pred = classify_with_llm(text, few_shot=few_shot)
    
    ensemble_result = {}
    for emotion in all_emotions:
        ensemble_result[emotion] = (
            weights["roberta"] * roberta_pred.get(emotion, 0) +
            weights["rule"] * rule_pred.get(emotion, 0) +
            weights["llm"] * llm_pred.get(emotion, 0)
        )
    return ensemble_result


# ============================================
# Method 2: Stacked Ensemble (Meta-learner)
# ============================================
class StackedEnsemble:
    """
    Stacked ensemble using a meta-learner to combine base model predictions.
    Meta-model learns optimal combination from validation data.
    """
    def __init__(self, meta_model_type='logistic'):
        self.meta_models = {}  # One meta-model per emotion
        self.meta_model_type = meta_model_type
        self.is_trained = False
    
    def _get_base_predictions(self, text, few_shot=True):
        """Get predictions from all base models"""
        roberta_pred = classify_with_roberta(text)
        rule_pred = rule_based_model.analyze_text(text)
        llm_pred = classify_with_llm(text, few_shot=few_shot)
        return roberta_pred, rule_pred, llm_pred
    
    def _create_meta_features(self, roberta_pred, rule_pred, llm_pred, emotion):
        """
        Create feature vector for meta-learner.
        Features: [roberta_score, rule_score, llm_score]
        Can be extended: Add confidence, text length, etc. features
        """
        features = [
            roberta_pred.get(emotion, 0),
            rule_pred.get(emotion, 0),
            llm_pred.get(emotion, 0)
        ]
        return np.array(features)
    
    def train(self, texts, labels, few_shot=True):
        """
        Train meta-models on validation set.
        
        Args:
            texts: list of text samples
            labels: list of dicts with ground truth labels
        """
        # Collect predictions from base models
        all_features = {emotion: [] for emotion in all_emotions}
        all_labels = {emotion: [] for emotion in all_emotions}
        
        print("Collecting base model predictions for stacking...")
        for i, (text, label) in enumerate(zip(texts, labels)):
            roberta_pred, rule_pred, llm_pred = self._get_base_predictions(text, few_shot)
            
            for emotion in all_emotions:
                features = self._create_meta_features(roberta_pred, rule_pred, llm_pred, emotion)
                all_features[emotion].append(features)
                # Binarize label at 0.5 threshold
                all_labels[emotion].append(1 if label.get(emotion, 0) >= 0.5 else 0)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(texts)} samples")
        
        # Train a meta-model for each emotion
        print("Training meta-models...")
        for emotion in all_emotions:
            X = np.array(all_features[emotion])
            y = np.array(all_labels[emotion])
            
            if self.meta_model_type == 'logistic':
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
            else:  # gradient boosting
                meta_model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                )
            
            meta_model.fit(X, y)
            self.meta_models[emotion] = meta_model
            print(f"  Trained meta-model for {emotion}")
        
        self.is_trained = True
        print("Stacked ensemble training complete.")
    
    def predict(self, text, few_shot=True):
        """Predict using trained meta-models"""
        if not self.is_trained:
            raise ValueError("Meta-models not trained. Call train() first.")
        
        roberta_pred, rule_pred, llm_pred = self._get_base_predictions(text, few_shot)
        
        result = {}
        for emotion in all_emotions:
            features = self._create_meta_features(roberta_pred, rule_pred, llm_pred, emotion)
            # Get probability of positive class
            prob = self.meta_models[emotion].predict_proba(features.reshape(1, -1))[0][1]
            result[emotion] = float(prob)
        
        return result
    
    def save(self, path):
        """Save trained meta-models"""
        with open(path, 'wb') as f:
            pickle.dump(self.meta_models, f)
    
    def load(self, path):
        """Load trained meta-models"""
        with open(path, 'rb') as f:
            self.meta_models = pickle.load(f)
        self.is_trained = True


# ============================================
# Method 3: Uncertainty-based Weighting
# ============================================
def get_roberta_uncertainty(text, n_samples=5):
    """
    Estimate RoBERTa uncertainty using MC Dropout.
    For simplicity, we use prediction variance as uncertainty proxy.
    """
    import torch
    from classify import roberta_model, roberta_tokenizer, device
    
    roberta_model.train()  # Enable dropout
    
    predictions = []
    encoding = roberta_tokenizer(
        text, padding="max_length", truncation=True, 
        max_length=512, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = roberta_model(**encoding).logits.sigmoid().cpu().numpy().flatten()
            predictions.append(output)
    
    roberta_model.eval()  # Disable dropout
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)  # Higher std = higher uncertainty
    
    return {emotion: float(mean_pred[i]) for i, emotion in enumerate(all_emotions)}, \
           {emotion: float(uncertainty[i]) for i, emotion in enumerate(all_emotions)}


def ensemble_uncertainty_weighted(text, few_shot=True, use_mc_dropout=False):
    """
    Uncertainty-based weighted ensemble.
    Models with higher uncertainty get lower weights.
    """
    # Get predictions
    if use_mc_dropout:
        roberta_pred, roberta_uncertainty = get_roberta_uncertainty(text)
    else:
        roberta_pred = classify_with_roberta(text)
        # Use prediction entropy as simple uncertainty proxy
        roberta_uncertainty = {
            emotion: abs(0.5 - roberta_pred.get(emotion, 0.5)) 
            for emotion in all_emotions
        }
        # Invert: closer to 0.5 = higher uncertainty
        roberta_uncertainty = {
            emotion: 1 - roberta_uncertainty[emotion] 
            for emotion in all_emotions
        }
    
    rule_pred = rule_based_model.analyze_text(text)
    llm_pred = classify_with_llm(text, few_shot=few_shot)
    
    # Simple uncertainty for rule-based (fixed low uncertainty for explicit matches)
    rule_uncertainty = {emotion: 0.3 for emotion in all_emotions}  # Generally confident
    
    # LLM uncertainty: use prediction confidence as inverse uncertainty
    llm_uncertainty = {
        emotion: 1 - abs(0.5 - llm_pred.get(emotion, 0.5))
        for emotion in all_emotions
    }
    
    # Calculate adaptive weights based on uncertainty (lower uncertainty = higher weight)
    result = {}
    for emotion in all_emotions:
        # Convert uncertainty to confidence (1 - uncertainty)
        conf_roberta = 1 - roberta_uncertainty.get(emotion, 0.5)
        conf_rule = 1 - rule_uncertainty.get(emotion, 0.5)
        conf_llm = 1 - llm_uncertainty.get(emotion, 0.5)
        
        # Normalize to get weights
        total_conf = conf_roberta + conf_rule + conf_llm + 1e-8  # Avoid division by zero
        
        w_roberta = conf_roberta / total_conf
        w_rule = conf_rule / total_conf
        w_llm = conf_llm / total_conf
        
        # Weighted combination
        result[emotion] = (
            w_roberta * roberta_pred.get(emotion, 0) +
            w_rule * rule_pred.get(emotion, 0) +
            w_llm * llm_pred.get(emotion, 0)
        )
    
    return result


# ============================================
# Unified evaluation interface
# ============================================
def evaluate_all_methods(test_texts, test_labels, stacked_model=None):
    """
    Evaluate all ensemble methods and compare performance.
    
    Returns:
        dict: Results for each method
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    results = {
        "static": {"predictions": [], "f1": {}},
        "stacked": {"predictions": [], "f1": {}},
        "uncertainty": {"predictions": [], "f1": {}},
        "roberta_only": {"predictions": [], "f1": {}},
        "llm_only": {"predictions": [], "f1": {}},
    }
    
    print("Evaluating all ensemble methods...")
    for i, (text, label) in enumerate(zip(test_texts, test_labels)):
        # Individual models
        roberta_pred = classify_with_roberta(text)
        llm_pred = classify_with_llm(text, few_shot=True)
        
        # Ensemble methods
        static_pred = ensemble_static(text)
        uncertainty_pred = ensemble_uncertainty_weighted(text)
        
        results["roberta_only"]["predictions"].append(roberta_pred)
        results["llm_only"]["predictions"].append(llm_pred)
        results["static"]["predictions"].append(static_pred)
        results["uncertainty"]["predictions"].append(uncertainty_pred)
        
        if stacked_model and stacked_model.is_trained:
            stacked_pred = stacked_model.predict(text)
            results["stacked"]["predictions"].append(stacked_pred)
        
        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(test_texts)} samples")
    
    # Calculate F1 scores for each emotion and method
    for method in results:
        if not results[method]["predictions"]:
            continue
            
        for emotion in all_emotions:
            y_true = [1 if label.get(emotion, 0) >= 0.5 else 0 for label in test_labels]
            y_pred = [1 if pred.get(emotion, 0) >= 0.5 else 0 
                     for pred in results[method]["predictions"]]
            
            results[method]["f1"][emotion] = f1_score(y_true, y_pred, zero_division=0)
        
        # Macro F1
        results[method]["macro_f1"] = np.mean(list(results[method]["f1"].values()))
    
    return results


def print_comparison_table(results):
    """Print comparison table for paper"""
    print("\n" + "="*70)
    print("COMPARISON OF ENSEMBLE METHODS")
    print("="*70)
    print(f"{'Method':<20} {'Macro F1':<12} " + " ".join([f"{e[:8]:<10}" for e in all_emotions]))
    print("-"*70)
    
    for method in ["roberta_only", "llm_only", "static", "stacked", "uncertainty"]:
        if method not in results or "macro_f1" not in results[method]:
            continue
        row = f"{method:<20} {results[method]['macro_f1']:.4f}      "
        for emotion in all_emotions:
            row += f"{results[method]['f1'].get(emotion, 0):.4f}     "
        print(row)
    
    print("="*70)


# ============================================
# Main: Training and evaluation example
# ============================================
if __name__ == "__main__":
    # Example data (replace with actual validation/test set)
    sample_val_data = [
        "I'm feeling hopeful but also nervous about the surgery.",
        "The doctor was very professional and I trust their judgment.",
        "I'm so frustrated with the long wait times at this clinic.",
        "After the treatment, I feel much more confident about my recovery.",
        "I'm anxious about the test results but trying to stay positive.",
    ]
    
    sample_val_labels = [
        {"confidence": 0.4, "satisfaction": 0.5, "hope": 0.8, "trust_medical": 0.6, "anxiety": 0.7, "anger": 0.1},
        {"confidence": 0.7, "satisfaction": 0.8, "hope": 0.6, "trust_medical": 0.9, "anxiety": 0.2, "anger": 0.0},
        {"confidence": 0.3, "satisfaction": 0.1, "hope": 0.2, "trust_medical": 0.4, "anxiety": 0.5, "anger": 0.8},
        {"confidence": 0.9, "satisfaction": 0.7, "hope": 0.8, "trust_medical": 0.7, "anxiety": 0.2, "anger": 0.0},
        {"confidence": 0.4, "satisfaction": 0.4, "hope": 0.6, "trust_medical": 0.5, "anxiety": 0.8, "anger": 0.1},
    ]
    
    # 1. Train Stacked Ensemble
    print("="*50)
    print("Training Stacked Ensemble...")
    print("="*50)
    stacked = StackedEnsemble(meta_model_type='logistic')
    stacked.train(sample_val_data, sample_val_labels)
    
    # 2. Evaluate all methods
    print("\n" + "="*50)
    print("Evaluating all methods...")
    print("="*50)
    results = evaluate_all_methods(sample_val_data, sample_val_labels, stacked)
    
    # 3. Print comparison table
    print_comparison_table(results)
    
    # 4. Save results
    output_file = "ensemble_comparison_results.json"
    try:
        with open(output_file, "w") as f:
            # Only save F1 scores, not predictions
            save_results = {
                method: {
                    "f1_scores": result["f1"],
                } for method, result in results.items()
            }
            json.dump(save_results, f, indent=4)
        
        # Get the absolute path for better user feedback
        abs_path = os.path.abspath(output_file)
        print(f"\nResults successfully saved to: {abs_path}")
        
    except IOError as e:
        print(f"\nError saving results to {output_file}: {str(e)}")
        print("Trying to save in the current working directory...")
        try:
            with open(os.path.join(os.getcwd(), output_file), "w") as f:
                json.dump(save_results, f, indent=4)
            print(f"Results saved to: {os.path.join(os.getcwd(), output_file)}")
        except Exception as e2:
            print(f"Failed to save results. Please check directory permissions. Error: {str(e2)}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while saving results: {str(e)}")