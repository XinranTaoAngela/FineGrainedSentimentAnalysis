import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from classify import classify_with_roberta, classify_with_llm
from ensemble import ensemble_prediction
from rule_based.rule_based import HealthcareSentimentDimensions

# Define emotion labels (must match those used in your models)
all_emotions = ["confidence", "satisfaction", "hope", "trust_medical", "anxiety", "anger"]

# Load test dataset (adjust the path if needed)
df_test = pd.read_csv("data/test_data.csv")
df_test=df_test.head(100)
# Convert the ground-truth 'emotions' string into a binary label dictionary
def parse_emotions(emotion_str):
    labels = [e.strip() for e in emotion_str.split(",")]
    return {emotion: (1 if emotion in labels else 0) for emotion in all_emotions}

df_test["label_vector"] = df_test["emotions"].apply(parse_emotions)
# Convert dictionary to ordered list for evaluation
y_true = df_test["label_vector"].apply(lambda d: [d[emotion] for emotion in all_emotions]).tolist()

# Instantiate the rule-based model once
rule_based_model = HealthcareSentimentDimensions()

# Helper function to threshold continuous predictions into binary labels
def threshold_predictions(pred_vector, threshold=0.5):
    return [1 if p >= threshold else 0 for p in pred_vector]

# Prepare lists to store predictions from each model
pred_roberta = []
pred_llm = []
pred_rule = []
pred_ensemble = []

print("Evaluating on {} test samples...".format(len(df_test)))
# Use tqdm to track progress over test samples
for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing test samples"):
    text = row["selftext"]

    # Get predictions from each model (each returns a dictionary with keys=all_emotions)
    roberta_pred = classify_with_roberta(text)
    llm_pred = classify_with_llm(text, few_shot=True)
    rule_pred = rule_based_model.analyze_text(text)
    ensemble_pred = ensemble_prediction(text, few_shot=True)

    # Use .get() with default value 0.0 to avoid KeyErrors in case any key is missing
    pred_roberta.append([roberta_pred.get(emotion, 0.0) for emotion in all_emotions])
    pred_llm.append([llm_pred.get(emotion, 0.0) for emotion in all_emotions])
    pred_rule.append([rule_pred.get(emotion, 0.0) for emotion in all_emotions])
    pred_ensemble.append([ensemble_pred.get(emotion, 0.0) for emotion in all_emotions])

# Threshold the predictions to obtain binary outputs
pred_roberta_bin = [threshold_predictions(pred) for pred in pred_roberta]
pred_llm_bin = [threshold_predictions(pred) for pred in pred_llm]
pred_rule_bin = [threshold_predictions(pred) for pred in pred_rule]
pred_ensemble_bin = [threshold_predictions(pred) for pred in pred_ensemble]

# Compute Macro F1 Scores for each method
f1_roberta = f1_score(y_true, pred_roberta_bin, average="macro")
f1_llm = f1_score(y_true, pred_llm_bin, average="macro")
f1_rule = f1_score(y_true, pred_rule_bin, average="macro")
f1_ensemble = f1_score(y_true, pred_ensemble_bin, average="macro")

# Prepare evaluation results dictionary
evaluation_results = {
    "Macro F1 Scores": {
        "RoBERTa": f1_roberta,
        "LLM": f1_llm,
        "Rule-based": f1_rule,
        "Ensemble": f1_ensemble
    },
    "Detailed Predictions": {
        "roberta": pred_roberta,
        "llm": pred_llm,
        "rule": pred_rule,
        "ensemble": pred_ensemble
    }
}

# Save the evaluation results to a new file
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("\nMacro F1 Scores:")
print("RoBERTa:   {:.4f}".format(f1_roberta))
print("LLM:       {:.4f}".format(f1_llm))
print("Rule-based:{:.4f}".format(f1_rule))
print("Ensemble:  {:.4f}".format(f1_ensemble))

print("\nEvaluation complete. Results saved to evaluation_results.json")
