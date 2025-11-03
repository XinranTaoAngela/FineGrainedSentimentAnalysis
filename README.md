# Fine-Grained Sentiment Analysis for Healthcare Reddit Posts

A hybrid system for fine-grained sentiment/emotion analysis on healthcare-related Reddit posts. It combines:

- Rule-based scoring using linguistic cues
- A fine-tuned RoBERTa sequence classifier
- An LLM classifier (zero-shot/few-shot)
- Weighted ensembling and evaluation utilities

The emotion dimensions tracked are: `confidence`, `satisfaction`, `hope`, `trust_medical`, `anxiety`, and `anger`.

## Features

- **Rule-based model** for interpretable, context-aware scoring (negation/intensifier handling)
- **RoBERTa classifier** loaded from a local fine-tuned checkpoint
- **LLM classification** with few-shot or zero-shot prompting
- **Ensemble** of rule-based, RoBERTa, and LLM predictions
- **Evaluation** scripts with Macro-F1 and prediction exports
- **Advanced LLM+BERT ensemble** with correlation analysis (optional)

## Repository Structure

- `classify.py`
  - Loads fine-tuned RoBERTa and provides `classify_with_roberta(text)`
  - LLM endpoint via `classify_with_llm(text, few_shot=False)`
- `ensemble.py`
  - `ensemble_prediction(text, few_shot=True)` combines rule, RoBERTa, and LLM outputs
- `evaluate.py`
  - Runs evaluation over `data/test_data.csv` and writes `evaluation_results.json`
- `healthcare_sentiment.py`
  - `HealthcareSentimentDimensions` rule-based analyzer and utilities
- `llm_healthcare_ensemble.py`
  - `LLMHealthcareEnsemble` advanced, async LLM+rule+BERT analysis with correlations
- `rule_based/`
  - `rule_based.py` backing implementation for rule-based analysis
- `bert/roberta_sentiment_model/`
  - HuggingFace model directory for the fine-tuned RoBERTa checkpoint
- `data/`
  - Datasets like `test_data.csv`, labeling and analysis notebooks
- Reports/figures: `confusion_matrix.pdf`, `cooccurrence_matrix.pdf`, `research_plots.pdf`, etc.

## Requirements

- Python 3.9+
- Recommended: GPU/CUDA (Linux/Windows) or Apple Silicon with MPS (macOS)

Python packages (install via pip):

- torch
- transformers
- scikit-learn
- pandas
- numpy
- tqdm
- spacy (model `en_core_web_sm` is auto-downloaded at runtime)
- networkx
- openai

Example installation:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch transformers scikit-learn pandas numpy tqdm spacy networkx openai
python -m spacy download en_core_web_sm
```

## Data

- Expected test file: `data/test_data.csv`
- For evaluation, the CSV should contain columns:
  - `selftext`: the Reddit post text
  - `emotions`: comma-separated labels from the set {confidence, satisfaction, hope, trust_medical, anxiety, anger}

Example row:

```
selftext,emotions
"I am hopeful but anxious about my upcoming surgery",hope,anxiety
```

Other large data files in `data/` (e.g., `RedditData.csv`, `Reddit_data.csv`) are raw corpora not required for the quick evaluation script, but useful for training/analysis notebooks.

## Model Checkpoints

- RoBERTa: expected at `bert/roberta_sentiment_model/` containing tokenizer and model files compatible with `transformers`.
- If you store the model elsewhere, update `roberta_path` in `classify.py` accordingly.

## API Keys and Security

- `classify.py` currently sets `openai.api_key` directly in code.
- Best practice is to set an environment variable and read it at runtime:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

And in code:

```python
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Do not commit real keys to version control.

## Usage

### 1) Single-text classification

Run the example in `classify.py` (prints RoBERTa and LLM predictions):

```bash
python classify.py
```

Programmatic usage:

```python
from classify import classify_with_roberta, classify_with_llm

text = "I'm hopeful but anxious about the MRI results."
roberta_pred = classify_with_roberta(text)
llm_pred = classify_with_llm(text, few_shot=True)
print(roberta_pred)
print(llm_pred)
```

### 2) Ensemble prediction

```bash
python ensemble.py
```

Programmatic usage:

```python
from ensemble import ensemble_prediction
scores = ensemble_prediction("Hospital staff was excellent though I still worry about costs.")
print(scores)
```

### 3) Evaluation on test set

- Expects `data/test_data.csv` with `selftext` and `emotions`
- Writes `evaluation_results.json`

```bash
python evaluate.py
```

### 4) Advanced LLM + BERT Ensemble (optional)

`llm_healthcare_ensemble.py` provides an async pipeline with per-dimension prompting and correlation analysis.

```bash
python llm_healthcare_ensemble.py
```

Or programmatically:

```python
import asyncio
from llm_healthcare_ensemble import LLMHealthcareEnsemble

async def run():
    analyzer = LLMHealthcareEnsemble(
        llm_model_name="gpt-3.5-turbo",
        bert_model_name="bert-base-uncased",
        use_few_shot=True,
    )
    results = await analyzer.analyze_with_correlation("Initially anxious, later confident after consulting the doctor.")
    print(results)

asyncio.run(run())
```

## Notes and Tips

- On macOS (Apple Silicon), the code will try to use MPS for PyTorch where applicable.
- If `transformers` or the spaCy model are missing, components may auto-download; ensure internet access or pre-download in your environment.
- Tune ensemble weights in `ensemble.py` and `llm_healthcare_ensemble.py` to suit your dataset or objectives.

## Outputs

- `test_results.json` from ensemble testing
- `evaluation_results.json` from evaluation
- Various PDFs with plots and analyses in the repository root: `confusion_matrix.pdf`, `cooccurrence_matrix.pdf`, `model_comparison.pdf`, `research_plots.pdf`, `research_scatter.pdf`, `text_analysis_plots.pdf`

## License

Specify your license here (e.g., MIT).
