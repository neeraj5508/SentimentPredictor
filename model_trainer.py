"""
Machine learning model training utilities for sentiment analysis
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

class SentimentModelTrainer:
    """
    Class for training sentiment analysis models
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        
    def create_vectorizer(self, max_features=5000, ngram_range=(1, 2)):
        """
        Create and configure TF-IDF vectorizer
        
        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
            
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        return vectorizer
    
    def create_model(self, random_state=42):
        """
        Create and configure Logistic Regression model
        
        Args:
            random_state (int): Random state for reproducibility
            
        Returns:
            LogisticRegression: Configured model
        """
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'
        )
        return model
    
    def train_model(self, texts, labels, test_size=0.2, random_state=42):
        """
        Train the sentiment analysis model
        
        Args:
            texts (list): List of text samples
            labels (list): List of sentiment labels (0: negative, 1: positive)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (model, vectorizer, accuracy, classification_report)
        """
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        if len(texts) < 4:
            raise ValueError("Need at least 4 samples to train the model")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Create and fit vectorizer
        self.vectorizer = self.create_vectorizer()
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Create and train model
        self.model = self.create_model(random_state)
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        self.is_trained = True
        
        return self.model, self.vectorizer, accuracy, report
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (prediction, probabilities)
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        
        return prediction, probabilities
    
    def get_feature_importance(self, top_n=10):
        """
        Get top features (words) that influence predictions
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Dictionary with positive and negative influential features
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained before extracting features")
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Get top positive and negative features
        positive_indices = np.argsort(coefficients)[-top_n:][::-1]
        negative_indices = np.argsort(coefficients)[:top_n]
        
        positive_features = [(feature_names[i], coefficients[i]) for i in positive_indices]
        negative_features = [(feature_names[i], coefficients[i]) for i in negative_indices]
        
        return {
            'positive_features': positive_features,
            'negative_features': negative_features
        }

# Convenience function for simple training
def train_sentiment_model(texts, labels, test_size=0.2, random_state=42):
    """
    Simple function to train a sentiment analysis model
    
    Args:
        texts (list): List of text samples
        labels (list): List of sentiment labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (model, vectorizer, accuracy)
    """
    trainer = SentimentModelTrainer()
    model, vectorizer, accuracy, report = trainer.train_model(
        texts, labels, test_size, random_state
    )
    
    return model, vectorizer, accuracy

def evaluate_model_performance(model, vectorizer, texts, labels):
    """
    Evaluate model performance on given data
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        texts (list): List of text samples
        labels (list): List of true labels
        
    Returns:
        dict: Dictionary with performance metrics
    """
    # Vectorize texts
    X = vectorizer.transform(texts)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }
