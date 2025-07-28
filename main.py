# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import time

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Custom CSS for better styling
st.set_page_config(
    page_title="🎬 Movie Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 3rem;
    }
    .positive-sentiment {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .negative-sentiment {
        background: linear-gradient(90deg, #F44336, #FF5722);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #FF6B6B;
        margin: 2rem 0;
    }
    .sample-review {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2196F3;
        margin: 0.5rem 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown('<h1 class="main-header">🎬 AI Movie Review Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover the sentiment behind movie reviews using advanced RNN technology</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("🤖 About the Model")
    st.info("""
    This app uses a Simple RNN (Recurrent Neural Network) trained on the IMDB dataset to classify movie reviews as positive or negative.
    
    **Model Details:**
    - Architecture: Simple RNN with Embedding
    - Dataset: IMDB Movie Reviews (50K reviews)
    - Accuracy: ~85%
    """)
    
    st.header("📊 Model Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", "25,000")
        st.metric("Vocabulary Size", "10,000")
    with col2:
        st.metric("Test Samples", "25,000")
        st.metric("Sequence Length", "500")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Enter Your Movie Review")
    
    # Sample reviews section
    st.write("**Try these sample reviews or write your own:**")
    
    sample_reviews = [
        "This movie was absolutely fantastic! Great acting and amazing plot.",
        "Terrible movie, waste of time. Poor acting and boring storyline.",
        "An okay film, nothing special but watchable on a weekend.",
        "Brilliant cinematography and outstanding performances by all actors!"
    ]
    
    selected_sample = st.selectbox("Choose a sample review:", [""] + sample_reviews)
    
    # Text area for user input
    if selected_sample:
        user_input = st.text_area('Movie Review', value=selected_sample, height=150, key="review_input")
    else:
        user_input = st.text_area('Movie Review', placeholder="Type your movie review here...", height=150, key="review_input")
    
    # Word count
    if user_input:
        word_count = len(user_input.split())
        st.caption(f"Word count: {word_count}")

with col2:
    st.subheader("🎯 Prediction Results")
    
    if st.button('🔍 Analyze Sentiment', type="primary", use_container_width=True):
        if user_input.strip():
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text('Preprocessing text...')
                elif i < 60:
                    status_text.text('Tokenizing words...')
                elif i < 90:
                    status_text.text('Running neural network...')
                else:
                    status_text.text('Generating prediction...')
                time.sleep(0.01)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Make prediction
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input, verbose=0)
            confidence = float(prediction[0][0])
            sentiment = 'Positive' if confidence > 0.5 else 'Negative'
            
            # Display results with styling
            if sentiment == 'Positive':
                st.markdown(f'<div class="positive-sentiment">😊 {sentiment}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="negative-sentiment">😞 {sentiment}</div>', unsafe_allow_html=True)
            
            # Confidence meter using Streamlit components
            st.subheader("📈 Confidence Score")
            
            # Visual confidence bar
            confidence_percent = confidence * 100
            if confidence > 0.5:
                st.success(f"Positive Confidence: {confidence_percent:.1f}%")
                st.progress(confidence)
            else:
                negative_confidence = (1 - confidence) * 100
                st.error(f"Negative Confidence: {negative_confidence:.1f}%")
                st.progress(1 - confidence)
            
            # Confidence visualization with colored metrics
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                if confidence > 0.7:
                    st.metric("Confidence Level", "High", delta="Strong Signal")
                elif confidence > 0.6 or confidence < 0.4:
                    st.metric("Confidence Level", "Medium", delta="Moderate Signal")
                else:
                    st.metric("Confidence Level", "Low", delta="Weak Signal")
            
            with col_conf2:
                st.metric("Probability", f"{confidence_percent:.1f}%")
            
            with col_conf3:
                certainty = abs(confidence - 0.5) * 2
                st.metric("Certainty", f"{certainty:.1%}")

        else:
            st.error("⚠️ Please enter a movie review to analyze!")
    else:
        st.info("👆 Click the button above to analyze your review!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎭 Built with Streamlit & TensorFlow | Powered by Simple RNN</p>
    <p>Made with ❤️ for movie enthusiasts</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎭 Built with Streamlit & TensorFlow | Powered by Simple RNN</p>
    <p>Made with ❤️ for movie enthusiasts</p>
</div>
""", unsafe_allow_html=True)

