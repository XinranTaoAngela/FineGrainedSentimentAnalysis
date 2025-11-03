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
import gensim

# **加载 spaCy 模型**
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

class HealthcareSentimentDimensions:
    """
    处理医疗相关文本情感分析，基于规则的方法 + word2vec + BERT
    """
    def __init__(self, word2vec_model=None):
        # **加载 BERT 模型**
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # **加载 spaCy 进行文本处理**
        self.nlp = nlp
        
        # **情感类别定义**
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

        # **强度修正词**
        self.intensity_modifiers = {
            'extremely': 2.0,
            'very': 1.5,
            'somewhat': 0.5,
            'slightly': 0.3,
            'barely': 0.2
        }

        # **加载 word2vec 模型**
        self.word2vec_model = word2vec_model

    def expand_emotion_dictionary(self, dimension, seed_words):
        """
        使用 word2vec 进行情感词扩展
        """
        if not self.word2vec_model:
            print("No word2vec model loaded. Returning original seed words.")
            return seed_words
        
        expanded_words = set(seed_words)
        for word in seed_words:
            # **gensim 现在使用 key_to_index 代替 vocab**
            if word in self.word2vec_model.key_to_index:
                similar_words = [w for w, _ in self.word2vec_model.most_similar(word, topn=5)]
                expanded_words.update(similar_words)
        return list(expanded_words)

    def context_aware_scoring(self, text, dimension):
        """
        计算文本在某个情感维度上的得分（考虑否定、增强词）
        """
        doc = self.nlp(text)
        dimension_score = 0

        for token in doc:
            # **匹配关键词**
            if token.text.lower() in self.emotion_dimensions[dimension]['keywords']:
                score = 1.0
                
                # **检查是否有否定词**
                if any(neg in [t.text.lower() for t in token.children] 
                       for neg in self.emotion_dimensions[dimension]['negators']):
                    score *= -1  # **否定词反转情感**

                # **检查是否有强度修正词**
                for child in token.children:
                    if child.text.lower() in self.intensity_modifiers:
                        score *= self.intensity_modifiers[child.text.lower()]

                dimension_score += score
                
        return dimension_score

    def analyze_text(self, text):
        """
        计算文本在所有情感维度上的得分，并返回归一化结果
        """
        scores = {dim: self.context_aware_scoring(text, dim) for dim in self.emotion_dimensions}
        
        # **归一化到 [0, 1] 区间**
        score_values = np.array(list(scores.values())).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_scores = scaler.fit_transform(score_values).flatten()
        
        return {dim: round(score, 2) for dim, score in zip(scores.keys(), normalized_scores)}

# **测试代码**
sentiment_analyzer = HealthcareSentimentDimensions()

text = "I'm very anxious about my surgery but I trust my doctor completely."
scores = sentiment_analyzer.analyze_text(text)
print(scores)  # 可能输出 {'anxiety': 0.9, 'trust_medical': 0.8, 'hope': 0.2}
