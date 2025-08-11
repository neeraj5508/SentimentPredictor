import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from data_handler import get_sample_dataset
from preprocessor import preprocess_text
from model_trainer import train_sentiment_model

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0

def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = 'sentiment_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            st.session_state.model = joblib.load(model_path)
            st.session_state.vectorizer = joblib.load(vectorizer_path)
            st.session_state.model_trained = True
            return True
        except:
            pass
    
    # Train new model if loading fails or files don't exist
    return train_new_model()

def train_new_model():
    """Train a new sentiment analysis model"""
    with st.spinner('Training sentiment analysis model...'):
        try:
            # Get dataset
            data = get_sample_dataset()
            df = pd.DataFrame(data)
            
            # Preprocess texts
            df['processed_text'] = df['text'].apply(preprocess_text)
            
            # Train model
            model, vectorizer, accuracy = train_sentiment_model(
                df['processed_text'].tolist(),
                df['sentiment'].tolist()
            )
            
            # Save model and vectorizer
            joblib.dump(model, 'sentiment_model.pkl')
            joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
            
            # Update session state
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.accuracy = accuracy
            st.session_state.model_trained = True
            
            return True
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False

def predict_sentiment(text):
    """Predict sentiment of input text"""
    if not st.session_state.model_trained:
        return None, None
    
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vector = st.session_state.vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = st.session_state.model.predict(text_vector)[0]
    probability = st.session_state.model.predict_proba(text_vector)[0]
    
    return prediction, probability

def main():
    st.title("🎭 Sentiment Analysis Application")
    st.markdown("**Analyze the sentiment of your text using Machine Learning**")
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if st.button("🔄 Retrain Model", help="Train a new model with the sample dataset"):
            st.session_state.model_trained = False
            if train_new_model():
                st.success("Model retrained successfully!")
                st.rerun()
            else:
                st.error("Failed to retrain model")
        
        if st.session_state.model_trained:
            st.success("✅ Model Ready")
            if hasattr(st.session_state, 'accuracy') and st.session_state.accuracy > 0:
                st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
        else:
            st.warning("⚠️ Model not trained")
        
        st.markdown("---")
        st.markdown("**Technology Stack:**")
        st.markdown("- Scikit-learn")
        st.markdown("- TF-IDF Vectorization")
        st.markdown("- Logistic Regression")
        st.markdown("- Streamlit")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input area
        user_input = st.text_area(
            "Enter your text for sentiment analysis:",
            height=150,
            placeholder="Type your text here... (e.g., 'I love this product!', 'This is terrible')"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_button or user_input:
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            elif not st.session_state.model_trained:
                st.error("❌ Model not trained. Please wait for training to complete.")
            else:
                # Make prediction
                prediction, probabilities = predict_sentiment(user_input)
                
                if prediction is not None:
                    # Display results
                    st.markdown("---")
                    st.header("📈 Analysis Results")
                    
                    # Sentiment result
                    sentiment_color = "green" if prediction == 1 else "red"
                    sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😟"
                    confidence = max(probabilities) * 100
                    
                    st.markdown(f"### Predicted Sentiment: <span style='color: {sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
                    
                    # Confidence metrics
                    col_neg, col_pos = st.columns(2)
                    with col_neg:
                        st.metric("Negative Probability", f"{probabilities[0]:.2%}")
                    with col_pos:
                        st.metric("Positive Probability", f"{probabilities[1]:.2%}")
                    
                    # Confidence bar
                    st.markdown("**Confidence Level:**")
                    st.progress(confidence / 100)
                    st.caption(f"Confidence: {confidence:.1f}%")
                    
                    # Interpretation
                    if confidence >= 80:
                        confidence_text = "Very confident"
                        confidence_icon = "🎯"
                    elif confidence >= 60:
                        confidence_text = "Moderately confident"
                        confidence_icon = "👍"
                    else:
                        confidence_text = "Low confidence"
                        confidence_icon = "🤔"
                    
                    st.info(f"{confidence_icon} {confidence_text} in this prediction")
    
    with col2:
        st.header("💡 Examples")
        st.markdown("**Try these examples:**")
        
        examples = [
            "I absolutely love this product!",
            "This movie is amazing and inspiring.",
            "What a terrible experience.",
            "I'm not happy with this service.",
            "It's okay, nothing special.",
            "Outstanding quality and great value!"
        ]
        
        for example in examples:
            if st.button(f"'{example[:30]}...'", key=f"example_{hash(example)}", help=example):
                st.session_state.example_text = example
                st.rerun()
        
        # Handle example text
        if 'example_text' in st.session_state:
            st.text_area("Selected Example:", st.session_state.example_text, key="example_display")
            if st.button("Analyze This Example"):
                if st.session_state.model_trained:
                    prediction, probabilities = predict_sentiment(st.session_state.example_text)
                    if prediction is not None:
                        sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😟"
                        confidence = max(probabilities) * 100
                        st.success(f"**{sentiment_label}** ({confidence:.1f}% confidence)")
                else:
                    st.error("Model not ready")
    
    # Dataset information
    st.markdown("---")
    with st.expander("📚 View Sample Dataset", expanded=False):
        data = get_sample_dataset()
        df = pd.DataFrame(data)
        df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})
        st.dataframe(df[['text', 'sentiment_label']], use_container_width=True)
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Positive Samples", len(df[df['sentiment'] == 1]))
        with col3:
            st.metric("Negative Samples", len(df[df['sentiment'] == 0]))

# Initialize the application
if __name__ == "__main__":
    # Load or train model on startup
    if not st.session_state.model_trained:
        load_or_train_model()
    
    main()
