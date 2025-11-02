import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import spacy
from collections import defaultdict
import networkx as nx

# load the English spaCy model 'en_core_web_sm'. 
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

class HealthcareSentimentDimensions:
    """
    Encapsulates methods and data structures for analyzing 
    healthcare-related sentiment across multiple dimensions (e.g., anxiety, anger).
    Uses a BERT model (from Hugging Face Transformers) for potential 
    classification tasks, and spaCy for text processing and token analysis.
    """
    def __init__(self):
        # Load pre-trained tokenizer and model from the Hugging Face library.
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Load spaCy English model for text processing and tokenization.
        self.nlp = spacy.load('en_core_web_sm')
        
        # Define multiple emotion dimensions with associated keywords, negators, and intensifiers.
        self.emotion_dimensions = {
            'confidence': {
                'keywords': ['confident', 'sure', 'certain', 'trust', 'reliable'],
                'negators': ['not', 'never', 'no', 'doubt'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            },
            'satisfaction': {
                'keywords': ['satisfied', 'happy', 'pleased', 'grateful', 'excellent'],
                'negators': ['not', 'never', 'no', 'dis'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            },
            'anxiety': {
                'keywords': ['worried', 'anxious', 'concerned', 'nervous', 'scared'],
                'negators': ['not', 'never', 'no'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            },
            'anger': {
                'keywords': ['angry', 'frustrated', 'upset', 'annoyed', 'mad'],
                'negators': ['not', 'never', 'no'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            },
            'hope': {
                'keywords': ['hopeful', 'optimistic', 'promising', 'looking forward'],
                'negators': ['not', 'never', 'no', 'less'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            },
            'trust_medical': {
                'keywords': ['trust doctor', 'professional', 'experienced', 'qualified'],
                'negators': ['not', 'never', 'no', 'dis'],
                'intensifiers': ['very', 'highly', 'extremely', 'completely']
            }
        }
        
        # Define a dictionary to handle intensity modifiers that scale emotion scores.
        self.intensity_modifiers = {
            'extremely': 2.0,
            'very': 1.5,
            'somewhat': 0.5,
            'slightly': 0.3,
            'barely': 0.2
        }
        
    def expand_emotion_dictionary(self, dimension, seed_words, word2vec_model):
        """
        Use a word2vec model to expand the emotion dictionary with words 
        similar to the seed words.
        
        :param dimension: The emotion dimension to expand.
        :param seed_words: Initial set of seed words for that dimension.
        :param word2vec_model: Trained word2vec model (e.g., from gensim) to find similar words.
        :return: List of expanded words.
        """
        expanded_words = set(seed_words)
        for word in seed_words:
            # Check if the word exists in the model's vocabulary.
            if word in word2vec_model.vocab:
                # Get top-5 most similar words from the word2vec model.
                similar_words = [w for w, _ in word2vec_model.most_similar(word, topn=5)]
                expanded_words.update(similar_words)
        return list(expanded_words)
    
    def context_aware_scoring(self, text, dimension):
        """
        Analyze the sentiment for a specific dimension in a piece of text, 
        accounting for negations and intensifiers.
        
        :param text: The input text to analyze.
        :param dimension: The emotion dimension to evaluate (e.g., "anxiety", "anger").
        :return: A single floating-point score for the specified dimension.
        """
        doc = self.nlp(text)
        dimension_score = 0
        
        # Iterate through each token in the text.
        for token in doc:
            # Check if the token is one of the dimension’s keywords.
            if token.text.lower() in self.emotion_dimensions[dimension]['keywords']:
                score = 1.0
                
                # Check if any child tokens are negators (e.g., "not", "never").
                if any(neg in [t.text.lower() for t in token.children] 
                       for neg in self.emotion_dimensions[dimension]['negators']):
                    score *= -1
                
                # If any child tokens are intensity modifiers, scale the score accordingly.
                for child in token.children:
                    if child.text.lower() in self.intensity_modifiers:
                        score *= self.intensity_modifiers[child.text.lower()]
                
                # Accumulate this token's contribution to the overall dimension score.
                dimension_score += score
                
        return dimension_score
    
    class MultiLabelEmotionClassifier(nn.Module):
        """
        A multi-label classifier built on top of a pre-trained BERT model 
        to handle multiple emotion dimensions simultaneously.
        """
        def __init__(self, num_dimensions):
            super().__init__()
            # Load a pre-trained BERT base model.
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            # Add dropout to reduce overfitting.
            self.dropout = nn.Dropout(0.1)
            # A linear layer to produce 'num_dimensions' outputs for multi-label classification.
            self.classifier = nn.Linear(768, num_dimensions)
            
        def forward(self, input_ids, attention_mask):
            # Get the BERT outputs. 
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            # The second element in `outputs` is typically the pooled output (CLS token).
            pooled_output = outputs[1]
            # Apply dropout.
            pooled_output = self.dropout(pooled_output)
            # Classify and apply sigmoid to get multi-label probabilities.
            logits = self.classifier(pooled_output)
            return torch.sigmoid(logits)
    
    def analyze_emotion_intensity(self, text):
        """
        Analyze and return the emotion intensity scores for each defined 
        dimension in a single text.
        
        :param text: The input text to be analyzed.
        :return: A dictionary mapping each dimension to its final intensity score.
        """
        dimension_intensities = {}
        
        for dimension in self.emotion_dimensions:
            # First, get the base context-aware score for the current dimension.
            base_score = self.context_aware_scoring(text, dimension)
            
            # We'll also re-check the entire text for intensity modifiers that apply more broadly.
            doc = self.nlp(text)
            intensity_multiplier = 1.0
            
            # If any tokens match the intensity modifiers, adjust the multiplier.
            for token in doc:
                if token.text.lower() in self.intensity_modifiers:
                    intensity_multiplier *= self.intensity_modifiers[token.text.lower()]
            
            # Multiply the base score by the aggregated intensity multiplier.
            final_score = base_score * intensity_multiplier
            dimension_intensities[dimension] = final_score
        
        return dimension_intensities
    
    def analyze_emotion_transitions(self, texts):
        """
        Given multiple texts (or multiple sentences), analyze how 
        the emotion scores change from one sentence to the next.
        
        :param texts: A list of text strings to be analyzed.
        :return: A list of lists, where each sub-list contains transition 
                 information between adjacent sentences.
        """
        emotion_sequences = []
        
        for text in texts:
            # Break the text into sentences using spaCy.
            sentences = self.nlp(text).sents
            sentence_emotions = []
            
            # Get emotion scores for each sentence.
            for sent in sentences:
                emotions = {}
                for dimension in self.emotion_dimensions:
                    score = self.context_aware_scoring(str(sent), dimension)
                    emotions[dimension] = score
                sentence_emotions.append(emotions)
            
            # Once we have per-sentence scores, compute transitions between them.
            transitions = []
            for i in range(len(sentence_emotions) - 1):
                transition = {
                    'from': sentence_emotions[i],
                    'to': sentence_emotions[i + 1],
                    'change': {
                        dim: sentence_emotions[i + 1][dim] - sentence_emotions[i][dim]
                        for dim in self.emotion_dimensions
                    }
                }
                transitions.append(transition)
            
            emotion_sequences.append(transitions)
        
        return emotion_sequences
    
    def analyze_emotion_correlations(self, texts):
        """
        Analyze the correlation between different emotion dimensions 
        across multiple input texts.
        
        :param texts: A list of text strings to analyze.
        :return: A pandas DataFrame representing the correlation matrix.
        """
        dimension_scores = defaultdict(list)
        
        # Collect scores for all dimensions across all texts.
        for text in texts:
            scores = self.analyze_emotion_intensity(text)
            for dimension, score in scores.items():
                dimension_scores[dimension].append(score)
        
        # Create a correlation matrix from the collected scores.
        dimensions = list(self.emotion_dimensions.keys())
        correlation_matrix = np.zeros((len(dimensions), len(dimensions)))
        
        for i, dim1 in enumerate(dimensions):
            for j, dim2 in enumerate(dimensions):
                # Compute correlation coefficient between pairs of emotion dimensions.
                correlation = np.corrcoef(
                    dimension_scores[dim1],
                    dimension_scores[dim2]
                )[0, 1]
                correlation_matrix[i, j] = correlation
        
        # Return a DataFrame with appropriate row/column labels.
        return pd.DataFrame(
            correlation_matrix,
            index=dimensions,
            columns=dimensions
        )
    
    def visualize_emotion_network(self, correlation_matrix):
        """
        Build and return a NetworkX graph representing the emotion dimension network. 
        Edges are created where correlation exceeds a threshold (e.g., 0.3).
        
        :param correlation_matrix: A pandas DataFrame of correlations between dimensions.
        :return: A NetworkX Graph object.
        """
        G = nx.Graph()
        
        # Retrieve dimension names from the correlation matrix index.
        dimensions = correlation_matrix.index
        for i, dim1 in enumerate(dimensions):
            G.add_node(dim1)
            for j, dim2 in enumerate(dimensions):
                if i < j:  # To avoid duplicate edges.
                    correlation = correlation_matrix.iloc[i, j]
                    # Add an edge if the absolute correlation is above a chosen threshold.
                    if abs(correlation) > 0.3:
                        G.add_edge(dim1, dim2, weight=abs(correlation))
        
        return G

# Example usage when running this script directly.
# Typically, you would import the class and use its methods in your application.
if __name__ == "__main__":
    # Instantiate the sentiment analysis class.
    analyzer = HealthcareSentimentDimensions()
    
    # Sample texts to demonstrate how the methods work.
    sample_texts = [
        "I'm very worried about the high cost of my treatment, but my doctor is extremely professional and caring.",
        "The hospital staff was helpful, though the waiting time was frustrating. I'm hopeful about my recovery.",
        "I can't afford these medical bills. It's making me extremely anxious and angry.",
        "The new treatment options look promising, and I trust my doctor's judgment completely."
    ]
    
    # Analyze emotion intensity for each sample text.
    for text in sample_texts:
        intensities = analyzer.analyze_emotion_intensity(text)
        print(f"\nText: {text}")
        print("Emotion intensities:")
        for dimension, score in intensities.items():
            print(f"{dimension}: {score:.2f}")
    
    # Analyze emotion transitions across sentences in each sample text.
    emotion_sequences = analyzer.analyze_emotion_transitions(sample_texts)
    print("\nEmotion transitions:")
    for i, sequence in enumerate(emotion_sequences):
        print(f"\nText {i + 1} transitions:")
        for transition in sequence:
            print(f"Changes: {transition['change']}")
    
    # Compute and display correlation matrix of emotions based on the sample texts.
    correlation_matrix = analyzer.analyze_emotion_correlations(sample_texts)
    print("\nEmotion correlations:")
    print(correlation_matrix)
