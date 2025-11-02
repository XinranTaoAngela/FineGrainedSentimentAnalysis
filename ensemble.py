import json
from classify import classify_with_roberta, classify_with_llm
from rule_based.rule_based import HealthcareSentimentDimensions  # Correct import

# Instantiate the rule-based model
rule_based_model = HealthcareSentimentDimensions()

# Ensemble function to combine predictions
def ensemble_prediction(text, few_shot=True):
    roberta_pred = classify_with_roberta(text)
    rule_pred = rule_based_model.analyze_text(text)  # Use analyze_text()
    llm_pred = classify_with_llm(text, few_shot=few_shot)

    # Define weights (you can tune these)
    weights = {"roberta": 0.5, "rule": 0.2, "llm": 0.3}

    ensemble_result = {emotion: (
        weights["roberta"] * roberta_pred.get(emotion, 0) +
        weights["rule"] * rule_pred.get(emotion, 0) +
        weights["llm"] * llm_pred.get(emotion, 0)
    ) for emotion in roberta_pred.keys()}

    return ensemble_result

# Evaluate all models
def evaluate_models(test_data, test_labels):
    results = {"roberta": [], "llm": [], "rule": [], "ensemble": []}

    for text, labels in zip(test_data, test_labels):
        roberta_pred = classify_with_roberta(text)
        llm_pred = classify_with_llm(text, few_shot=True)  # ✅ Only call once
        rule_pred = rule_based_model.analyze_text(text)

        results["roberta"].append(roberta_pred)
        results["llm"].append(llm_pred)
        results["rule"].append(rule_pred)
        results["ensemble"].append(ensemble_prediction(text))

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved to `test_results.json`")

# Example test set evaluation
if __name__ == "__main__":
    sample_test_data = ["I'm feeling hopeful but also nervous about the medical in usa."]
    sample_test_labels = [{"confidence": 0.6, "satisfaction": 0.5, "hope": 1.0, "trust_medical": 0.8, "anxiety": 0.4, "anger": 0.0}]
    
    evaluate_models(sample_test_data, sample_test_labels)
