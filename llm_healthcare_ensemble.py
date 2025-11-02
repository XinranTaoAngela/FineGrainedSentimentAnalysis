try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not found. Some features will be disabled.")
    SKLEARN_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional
import json

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("transformers not found. Some features will be disabled.")
    TRANSFORMERS_AVAILABLE = False

# Import the base class from the original module
from healthcare_sentiment import HealthcareSentimentDimensions

class BERTClassifier(nn.Module):
    """BERT-based classifier for emotion dimensions"""
    def __init__(self, num_labels: int, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
    def forward(self, **inputs):
        return self.bert(**inputs)

class LLMHealthcareEnsemble(HealthcareSentimentDimensions):
    """
    Enhanced healthcare sentiment analysis with LLM integration and ensemble methods.
    Implements both zero-shot and few-shot approaches, combining multiple models.
    """
    def __init__(self, 
                 llm_model_name: str = "gpt-4o",
                 bert_model_name: str = "bert-base-uncased",
                 use_few_shot: bool = True,
                 ensemble_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Check if required dependencies are available
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("This class requires the 'transformers' package. Please install it first.")
        
        self.llm_model_name = llm_model_name
        self.bert_model_name = bert_model_name
        self.use_few_shot = use_few_shot
        
        # Initialize BERT components
        self._initialize_bert_classifier()
        
        # Default ensemble weights if none provided
        self.ensemble_weights = ensemble_weights or {
            'rule_based': 0.3,
            'bert': 0.3,
            'llm': 0.4
        }
        
        # Initialize few-shot examples for each dimension
        self.few_shot_examples = {
            'anxiety': [
                ("I'm really worried about my upcoming surgery next week.", 0.8),
                ("The doctor explained everything clearly, but I still have concerns.", 0.6),
                ("Everything seems fine with my treatment.", 0.2)
            ],
            'satisfaction': [
                ("The medical staff was excellent and very attentive.", 0.9),
                ("The treatment went okay, though there were some issues.", 0.5),
                ("I'm disappointed with the level of care I received.", 0.2)
            ],
            'trust_medical': [
                ("My doctor is highly qualified and explains everything thoroughly.", 0.9),
                ("The medical team seems competent but communication could be better.", 0.6),
                ("I'm not sure if I'm getting the best possible care.", 0.3)
            ],
            'financial_stress': [
                ("These medical bills are overwhelming and I can't afford them.", 0.9),
                ("The treatment is expensive but insurance covers most of it.", 0.5),
                ("The costs are very reasonable and within my budget.", 0.2)
            ],
            'hope': [
                ("The new treatment is showing promising results already.", 0.9),
                ("There's been some improvement but it's slow progress.", 0.6),
                ("I'm not seeing much change in my condition.", 0.3)
            ]
        }
        
        # Initialize prompt templates
        self._initialize_prompts()
        
        # Initialize LLM client
        self._initialize_llm_client()

    def _initialize_bert_classifier(self):
        """Initialize BERT tokenizer and classifier"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        num_dimensions = len(self.emotion_dimensions)
        self.bert_classifier = BERTClassifier(num_dimensions, self.bert_model_name)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_classifier.to(self.device)
        self.bert_classifier.eval()  # Set to evaluation mode

    def _initialize_prompts(self):
        """Initialize prompt templates"""
        self.zero_shot_prompt_template = """
        Analyze the following healthcare-related text for the emotion dimension of {dimension}.
        Score the intensity from 0.0 (none) to 1.0 (very high).
        
        Text: {text}
        
        Provide only the numerical score as output.
        """
        
        self.few_shot_prompt_template = """
        Analyze healthcare-related texts for the emotion dimension of {dimension}.
        Score the intensity from 0.0 (none) to 1.0 (very high).
        
        Examples:
        {examples}
        
        Now analyze this text:
        {text}
        
        Provide only the numerical score as output.
        """

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        if 'gpt' in self.llm_model_name.lower():
            try:
                import openai
                self.llm_client = openai
            except ImportError:
                raise ImportError("OpenAI package not found. Please install it with 'pip install openai'")
        elif 'claude' in self.llm_model_name.lower():
            try:
                import anthropic
                self.llm_client = anthropic.Client()
            except ImportError:
                raise ImportError("Anthropic package not found. Please install it with 'pip install anthropic'")

    def get_bert_predictions(self, text: str) -> Dict[str, float]:
        """
        Get predictions from BERT model for all emotion dimensions
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping emotion dimensions to their predicted scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.bert_classifier(**inputs)
            logits = outputs.logits
            scores = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Create dictionary of predictions
        predictions = {}
        for dim, score in zip(self.emotion_dimensions.keys(), scores):
            predictions[dim] = float(score)
            
        return predictions

    async def get_llm_response(self, prompt: str) -> float:
        """Get response from LLM API"""
        try:
            if 'gpt' in self.llm_model_name.lower():
                response = await self.llm_client.ChatCompletion.acreate(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                score = float(response.choices[0].message.content.strip())
            elif 'claude' in self.llm_model_name.lower():
                response = await self.llm_client.messages.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                score = float(response.content[0].text.strip())
            
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return 0.5  # Return neutral score on error

    def prepare_few_shot_examples(self, dimension: str) -> str:
        """Format few-shot examples for the prompt"""
        examples = self.few_shot_examples.get(dimension, [])
        formatted_examples = ""
        for text, score in examples:
            formatted_examples += f"Text: {text}\nScore: {score}\n\n"
        return formatted_examples

    async def analyze_dimension_llm(self, text: str, dimension: str) -> float:
        """Analyze a single dimension using LLM"""
        if self.use_few_shot:
            examples = self.prepare_few_shot_examples(dimension)
            prompt = self.few_shot_prompt_template.format(
                dimension=dimension,
                examples=examples,
                text=text
            )
        else:
            prompt = self.zero_shot_prompt_template.format(
                dimension=dimension,
                text=text
            )
            
        return await self.get_llm_response(prompt)

    async def analyze_all_dimensions_llm(self, text: str) -> Dict[str, float]:
        """Analyze all dimensions using LLM"""
        results = {}
        for dimension in self.emotion_dimensions.keys():
            score = await self.analyze_dimension_llm(text, dimension)
            results[dimension] = score
        return results

    def ensemble_predictions(self, text: str, llm_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions from all methods"""
        rule_based_scores = self.analyze_emotion_intensity(text)
        bert_scores = self.get_bert_predictions(text)
        
        final_scores = {}
        for dimension in self.emotion_dimensions.keys():
            weighted_score = (
                self.ensemble_weights['rule_based'] * rule_based_scores[dimension] +
                self.ensemble_weights['bert'] * bert_scores[dimension] +
                self.ensemble_weights['llm'] * llm_scores[dimension]
            )
            final_scores[dimension] = weighted_score
            
        return final_scores

    async def analyze_with_correlation(self, text: str) -> Dict[str, Union[float, Dict]]:
        """Perform comprehensive analysis including correlations"""
        rule_based = self.analyze_emotion_intensity(text)
        bert = self.get_bert_predictions(text)
        llm = await self.analyze_all_dimensions_llm(text)
        
        methods = ['rule_based', 'bert', 'llm']
        scores = [rule_based, bert, llm]
        correlation_matrix = np.zeros((3, 3))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i <= j:
                    correlation = np.corrcoef(
                        list(scores[i].values()),
                        list(scores[j].values())
                    )[0, 1]
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
        
        ensemble = self.ensemble_predictions(text, llm)
        
        return {
            'ensemble_scores': ensemble,
            'individual_scores': {
                'rule_based': rule_based,
                'bert': bert,
                'llm': llm
            },
            'method_correlations': {
                'matrix': correlation_matrix.tolist(),
                'methods': methods
            }
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        analyzer = LLMHealthcareEnsemble(
            llm_model_name="gpt-3.5-turbo",
            bert_model_name="bert-base-uncased",
            use_few_shot=True
        )
        
        sample_text = """
        Initially very anxious about my diagnosis, but after discussing treatment 
        options with my doctor, I felt more confident. The medication costs are 
        concerning, but the hospital staff has been extremely supportive in helping 
        me understand my insurance coverage.
        """
        
        results = await analyzer.analyze_with_correlation(sample_text)
        
        print("\nEnsemble Scores:")
        for dimension, score in results['ensemble_scores'].items():
            print(f"{dimension}: {score:.2f}")
            
        print("\nMethod Correlations:")
        for i, method1 in enumerate(results['method_correlations']['methods']):
            for j, method2 in enumerate(results['method_correlations']['methods']):
                corr = results['method_correlations']['matrix'][i][j]
                print(f"{method1} vs {method2}: {corr:.2f}")
    
    asyncio.run(main())