# Overview

This is a sentiment analysis web application built with Streamlit that provides real-time text classification. The application uses machine learning techniques including TF-IDF vectorization and Logistic Regression to classify text as positive or negative with confidence scores. It features a complete ML pipeline with preprocessing, training, and prediction capabilities, along with an interactive web interface for user-friendly text input and results display.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Single-page web application with interactive widgets for text input and results display
- **Session State Management**: Uses Streamlit's session state to maintain model training status, loaded models, and accuracy metrics across user interactions
- **Page Configuration**: Configured with custom title, icon, and wide layout for better user experience

## Backend Architecture
- **Modular Design**: Separated into distinct modules for different responsibilities:
  - `app.py`: Main application controller and UI logic
  - `data_handler.py`: Sample dataset management and data provision
  - `model_trainer.py`: Machine learning model training utilities
  - `preprocessor.py`: Text preprocessing and cleaning functions

## Machine Learning Pipeline
- **Text Preprocessing**: Custom text cleaning pipeline that converts to lowercase, removes punctuation, numbers, and extra whitespace
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with configurable parameters including max features, n-gram ranges, and English stop words removal
- **Classification Model**: Logistic Regression algorithm for binary sentiment classification (positive/negative)
- **Model Persistence**: Trained models and vectorizers are saved using joblib for reuse across sessions

## Data Management
- **Embedded Dataset**: Self-contained sample dataset with balanced positive and negative sentiment examples
- **Train/Test Split**: Uses scikit-learn's train_test_split for model validation
- **Data Structure**: Simple dictionary format with 'text' and 'sentiment' keys for easy processing

## Application Flow
- **Model Loading/Training**: Attempts to load pre-trained models first, falls back to training new models if needed
- **Real-time Prediction**: Processes user input through the preprocessing pipeline, vectorizes text, and returns sentiment predictions with confidence scores
- **Performance Metrics**: Displays model accuracy and provides classification reports for model evaluation

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for creating the interactive user interface
- **scikit-learn**: Machine learning library providing TF-IDF vectorization, Logistic Regression, and evaluation metrics
- **pandas**: Data manipulation and analysis library for handling datasets
- **numpy**: Numerical computing library for array operations
- **joblib**: Model serialization and persistence library for saving/loading trained models

## Python Standard Library
- **os**: File system operations for checking model file existence
- **re**: Regular expressions for text preprocessing
- **string**: String operations and punctuation handling
- **warnings**: Managing and suppressing library warnings

## Model Persistence
- **File System Storage**: Models and vectorizers are saved as .pkl files in the local directory
- **Automatic Loading**: Application automatically detects and loads existing models on startup
- **Fallback Training**: If model files don't exist or fail to load, the system automatically trains new models