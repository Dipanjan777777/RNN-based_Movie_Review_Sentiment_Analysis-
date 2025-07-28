# 🎬 ReelFeel - AI Movie Review Sentiment Analyzer

An end-to-end deep learning project that uses Simple RNN (Recurrent Neural Network) to analyze movie review sentiments and classify them as positive or negative.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Approach](#-approach)
- [Architecture](#-architecture)
- [Environment Setup](#-environment-setup)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Demo](#-demo)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Problem Statement

Movie reviews are abundant on platforms like IMDB, but manually analyzing sentiment from thousands of reviews is time-consuming and subjective. This project aims to:

- **Automate sentiment analysis** of movie reviews
- **Classify reviews** as positive or negative with confidence scores
- **Provide real-time predictions** through an interactive web interface
- **Demonstrate RNN capabilities** for sequential text processing

### Business Impact
- Help movie studios gauge audience reception
- Assist viewers in making informed movie choices
- Enable automated content moderation for review platforms
- Provide insights for marketing and promotional strategies

## 🚀 Approach

### 1. **Data Preprocessing**
- Load IMDB dataset with 50,000 movie reviews
- Tokenize and encode text using word indices
- Pad sequences to uniform length (500 tokens)
- Split data into training (25K) and testing (25K) sets

### 2. **Model Architecture**
- **Embedding Layer**: Converts word indices to dense vectors (128 dimensions)
- **Simple RNN Layer**: Processes sequential information (128 units, tanh activation)
- **Dense Output Layer**: Binary classification with sigmoid activation

### 3. **Training Strategy**
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Early Stopping: Prevents overfitting
- Validation Split: 20% for monitoring

### 4. **Deployment**
- Interactive Streamlit web application
- Real-time sentiment prediction
- Confidence visualization
- Sample reviews for testing

## 🏗️ Architecture

```
Input Text → Tokenization → Padding → Embedding → Simple RNN → Dense → Sentiment Output
    ↓              ↓           ↓          ↓           ↓         ↓           ↓
"Great movie"  → [1,2,3]   → [0,0,1,2,3] → Dense    → Hidden  → Binary   → Positive
                                         Vectors     States    Class     (0.85)
```

### Model Summary
```
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 500, 128)         1,280,000
simple_rnn (SimpleRNN)      (None, 128)               32,896    
dense (Dense)               (None, 1)                 129       
=================================================================
Total params: 1,313,025
Trainable params: 1,313,025
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8 or higher
- UV package manager (recommended) or pip

### Option 1: Using UV (Recommended)

1. **Install UV** (if not already installed):
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone https://github.com/yourusername/reelfeel.git
cd reelfeel
```

3. **Create and activate virtual environment**:
```bash
uv venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

### Option 2: Using Traditional pip

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/reelfeel.git
cd reelfeel
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

## 📊 Dataset

**Source**: IMDB Movie Reviews Dataset (built into TensorFlow/Keras)

### Dataset Statistics
- **Total Reviews**: 50,000
- **Training Set**: 25,000 reviews
- **Test Set**: 25,000 reviews
- **Classes**: Binary (Positive/Negative)
- **Vocabulary Size**: 10,000 most frequent words
- **Average Review Length**: ~200 words
- **Max Sequence Length**: 500 tokens (padded)

### Data Distribution
- **Positive Reviews**: 50% (25,000)
- **Negative Reviews**: 50% (25,000)
- **Balanced Dataset**: No class imbalance issues

## 📈 Model Performance

### Training Results
- **Final Training Accuracy**: ~90%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~85%
- **Training Time**: ~10-15 minutes (CPU)
- **Model Size**: ~5MB

### Performance Metrics
```
              Precision  Recall  F1-Score  Support
Negative         0.84     0.86     0.85    12500
Positive         0.86     0.84     0.85    12500

Accuracy                           0.85    25000
Macro Avg        0.85     0.85     0.85    25000
```

## 🎮 Usage

### 1. Train the Model
```bash
# Open and run the training notebook
jupyter notebook simpleRNN.ipynb
# Or run cell-by-cell in VS Code
```

### 2. Test Predictions
```bash
# Open and run the prediction notebook
jupyter notebook prediction.ipynb
```

### 3. Launch Web Application
```bash
streamlit run main.py
```

### 4. Use the Web Interface
1. Open browser to `http://localhost:8501`
2. Enter a movie review or select a sample
3. Click "🔍 Analyze Sentiment"
4. View results with confidence scores

### 5. Command Line Prediction
```python
from tensorflow.keras.models import load_model
# Load model and make predictions (see prediction.ipynb)
```

## 📁 Project Structure

```
reelfeel/
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 main.py                   # Streamlit web application
├── 📓 simpleRNN.ipynb          # Model training notebook
├── 📓 prediction.ipynb         # Testing and prediction notebook
├── 🤖 simple_rnn_imdb.h5      # Trained model file
├── 📁 .venv/                   # Virtual environment (created after setup)
├── 📁 __pycache__/            # Python cache files
└── 📁 .streamlit/             # Streamlit configuration (optional)
```

## ✨ Features

### 🖥️ Web Application
- **Modern UI**: Clean, responsive design with custom CSS
- **Real-time Analysis**: Instant sentiment prediction
- **Progress Visualization**: Loading animations and progress bars
- **Confidence Metrics**: Detailed confidence scores and certainty levels
- **Sample Reviews**: Pre-loaded examples for quick testing
- **Interactive Elements**: Hover effects and smooth transitions

### 🤖 Model Features
- **Simple RNN Architecture**: Easy to understand and modify
- **Embedding Layer**: Efficient word representation
- **Early Stopping**: Prevents overfitting during training
- **Binary Classification**: Clear positive/negative sentiment
- **Confidence Scores**: Probability estimates for predictions

### 📱 User Experience
- **Responsive Design**: Works on desktop and mobile
- **Intuitive Interface**: Easy-to-use text input and buttons
- **Visual Feedback**: Color-coded results and metrics
- **Educational Sidebar**: Model information and statistics

## 🎥 Demo

### Sample Predictions

**Positive Review Example**:
```
Input: "This movie was absolutely fantastic! Great acting and amazing plot."
Output: 😊 Positive (Confidence: 89.2%)
```

**Negative Review Example**:
```
Input: "Terrible movie, waste of time. Poor acting and boring storyline."
Output: 😞 Negative (Confidence: 91.7%)
```

### Screenshots
*(Add screenshots of your web application here)*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/reelfeel.git
cd reelfeel

# Set up environment
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Start development server
streamlit run main.py
```

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**:
```bash
# Make sure to train the model first
jupyter notebook simpleRNN.ipynb
# Run all cells to generate simple_rnn_imdb.h5
```

**2. Import errors**:
```bash
# Verify virtual environment is activated
which python  # Should point to .venv/bin/python
pip list  # Check installed packages
```

**3. Streamlit port issues**:
```bash
# Try different port
streamlit run main.py --server.port 8502
```

**4. Memory issues during training**:
```python
# Reduce batch size in training notebook
batch_size = 16  # Instead of 32
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras Team** for the IMDB dataset and deep learning framework
- **Streamlit Team** for the amazing web app framework
- **IMDB** for providing the movie review dataset
- **Open Source Community** for continuous inspiration and support

## 📞 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---

**Made with ❤️ for the AI and movie enthusiast community**

*Last updated: December 2024*
