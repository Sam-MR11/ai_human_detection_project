# STREAMLIT APP FOR AI Vs Human Text Classification
# =====================================================
import pickle
import re
import string
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# NLTK for text processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import docx
import PyPDF2

# Scikit-learn for machine learning pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# File processing libraries
from PyPDF2 import PdfReader

# ============================================================================
# 2. NLTK DOWNLOADS
# ============================================================================
# Using a function with st.cache_resource to prevent re-downloading on every script run.
@st.cache_resource
def download_nltk_data():
    """downloads NLTK resources if they are not found."""
    try:
        nltk.data.find('corpora/wordnet.zip')
    except:
        nltk.download('wordnet', force=True)
    try:
        nltk.data.find('corpora/stopwords.zip')
    except:
        nltk.download('stopwords', force=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', force=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except:
        nltk.download('punkt_tab', force=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    except:
        nltk.download('averaged_perceptron_tagger', force=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng.zip')
    except:
        nltk.download('averaged_perceptron_tagger_eng', force=True)


download_nltk_data()

# Page Configuration
st.set_page_config(
    page_title="AI Vs Human Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082; /* Indigo */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333333;
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
    }
    .stButton>button:hover {
        background-color: #6A5ACD; /* SlateBlue */
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 6px solid #4B0082;
    }
</style>
""", unsafe_allow_html=True)

# # ============================================================================
# # CUSTOM PREPROCESSOR CLASS
# # ============================================================================
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer_1 = WordNetLemmatizer()
        self.stop_words.update(['u','didn,äôt', 'ü', 'ur', '4', '2', 'im','äôt','ä', 'dont','doesnt', 'doin','doesn,äôt', 'ure', 'ô'])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # The pipeline will pass a list of documents (or a single one in a list)
        if isinstance(X, (list, pd.Series)):
            return [self.preprocess(text) for text in X]
        elif isinstance(X, str):
            return self.preprocess(X)
        else:
            return X

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                     "N": wordnet.NOUN,
                       "V": wordnet.VERB,
                         "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def text_process(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def stopwords_removal(self, text):
        return ' '.join([word for word in text.split() if word not in self.stop_words and len(word) > 1])

    def lemmatizer(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer_1.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens if len(token) > 2]
        return ' '.join(lemmatized_tokens)

    def preprocess(self, text):
        cleaned_text = self.text_process(text)
        lemmatized_text = self.lemmatizer(self.stopwords_removal(cleaned_text))
        return lemmatized_text
# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
@st.cache_resource
def load_models():
    """
    Loads all models. The main SVM pipeline is loaded first, and its fitted
    vectorizer is extracted and used for the other individual models.
    This fixes the 'not fitted' error.
    """
    models = {}
    model_dir = 'models/' # models are in 'models' sub-folder

    # Load SVM pipeline (full pipeline)
    try:
        models['pipeline'] = joblib.load(os.path.join(model_dir, "Human_Vs_AI_Written_pipeline.pkl"))
        models['pipeline_available'] = True
        # Extract fitted vectorizer from pipeline for other models to reuse
        models['vectorizer'] = models['pipeline'].named_steps['tfidf']
        models['vectorizer_available'] = True
    except FileNotFoundError:
        models['pipeline_available'] = False
        models['vectorizer_available'] = False

    # Load individual SVM model and vectorizer if not loaded already
    if not models['vectorizer_available']:
        try:
            models['vectorizer'] = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False
    
    try:
        models['SVM'] = joblib.load(os.path.join(model_dir, "optimized_svm_model.pkl"))
        models['svm_available'] = True
    except FileNotFoundError:
        models['svm_available'] = False

    # Load Decision Tree pipeline 
    try:
        models['Decision Tree'] = joblib.load(os.path.join(model_dir, "decision_tree_pipeline.pkl"))
        models['dt_available'] = True
    except FileNotFoundError:
        models['dt_available'] = False

    # Load AdaBoost pipeline
    try:
        models['AdaBoost'] = joblib.load(os.path.join(model_dir, "adaboost_pipeline.pkl"))
        models['ada_available'] = True
    except FileNotFoundError:
        models['ada_available'] = False
    
    # Make sure at least one model setup is available
    if not (models['pipeline_available'] or (models['vectorizer_available'] and (models['svm_available'] or models['dt_available'] or models['ada_available']))):
        st.error("No valid model pipelines or individual models found! Check your 'models' folder.")
        return None

    return models

    
# ============================================================================
# FUNCTIONS FOR FILE PROCESSING
# ============================================================================

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text).strip()
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_name, models):
    """
    Make prediction using either:
    - Full pipeline (for SVM, Decision Tree, AdaBoost pipelines)
    - Individual model + vectorizer
    """
    try:
        class_names = ['Human-Written', 'AI-Generated']

        # Handle SVM pipeline (full pipeline)
        if model_name == "SVM" and models.get('pipeline_available', False):
            pipeline = models.get('pipeline')
            if pipeline is None:
                st.error("SVM pipeline not loaded.")
                return None, None
            prediction = pipeline.predict([text])[0]
            probabilities = pipeline.predict_proba([text])[0]

        # Handle Decision Tree pipeline
        elif model_name == "Decision Tree" and models.get('dt_available', False):
            dt_pipeline = models.get('Decision Tree')
            if dt_pipeline is None:
                st.error("Decision Tree pipeline not loaded.")
                return None, None
            prediction = dt_pipeline.predict([text])[0]
            probabilities = dt_pipeline.predict_proba([text])[0]

        # Handle AdaBoost pipeline
        elif model_name == "AdaBoost" and models.get('ada_available', False):
            ada_pipeline = models.get('AdaBoost')
            if ada_pipeline is None:
                st.error("AdaBoost pipeline not loaded.")
                return None, None
            prediction = ada_pipeline.predict([text])[0]
            probabilities = ada_pipeline.predict_proba([text])[0]

        # Fallback: individual models + vectorizer
        else:
            vectorizer = models.get('vectorizer')
            model = models.get(model_name)
            if vectorizer is None or model is None:
                st.error(f"Model or vectorizer not available for {model_name}.")
                return None, None
            
            # Preprocess text before vectorizing
            preprocessor = TextPreprocessor()
            processed_text = preprocessor.transform(text)
            features = vectorizer.transform([processed_text])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

        prob_dict = {
            'Human-Written': probabilities[0],
            'AI-Generated': probabilities[1]
        }
        return class_names[prediction], prob_dict

    except Exception as e:
        st.error(f"Prediction error for {model_name}: {e}")
        return None, None

def get_available_models(models):
    """
    Returns list of tuples (model_key, display_name)
    Only includes models for which pipelines or vectorizer+model are loaded.
    """
    available = []

    # If full SVM pipeline is loaded
    if models.get('pipeline_available', False):
        available.append(("SVM", "🤖 Support Vector Machine (Pipeline)"))
    else:
        # If individual SVM model and vectorizer are available
        if models.get('svm_available', False) and models.get('vectorizer_available', False):
            available.append(("SVM", "🤖 Support Vector Machine (Individual)"))

    # Decision Tree pipeline available
    if models.get('dt_available', False):
        available.append(("Decision Tree", "🌳 Decision Tree"))

    # AdaBoost pipeline available
    if models.get('ada_available', False):
        available.append(("AdaBoost", "🚀 AdaBoost"))

    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Real-Time Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.markdown("Cross-validation accuracy scores:")

# Model accuracy scores (update with your actual scores)
model_scores = {
    "SVM": 0.8149,
    "Decision Tree": 0.7975,
    "AdaBoost": 0.8112
}

fig_perf, ax_perf = plt.subplots()
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), ax=ax_perf, palette='coolwarm')
ax_perf.set_title("Model Accuracy Comparison")
ax_perf.set_ylabel("Mean CV Accuracy")
ax_perf.set_ylim(0.78, 0.83)
st.sidebar.pyplot(fig_perf)

st.sidebar.info("This app uses ML models trained on essays to detect AI-generated content.")

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI vs Human Text Detection application. This tool leverages machine learning to distinguish between text written by a human and text generated by an AI.
    
    **This project allows you to:**
    - Analyze text from various sources (direct input, `.txt`, `.docx`, `.pdf`).
    - Choose from three different machine learning models (SVM, Decision Tree, AdaBoost).
    - Receive real-time predictions with confidence scores.
    - Compare model performance side-by-side.
    """)
    
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if models.get('svm_available'):
                st.info("**🤖 SVM**\n✅ Ready")
            else:
                st.warning("**🤖 SVM**\n❌ Not Available")
        
        with col2:
            if models.get('dt_available'):
                st.info("**🌳 Decision Tree**\n✅ Ready")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")

        with col3:
            if models.get('ada_available'):
                st.info("**🚀 AdaBoost**\n✅ Ready")
            else:
                st.warning("**🚀 AdaBoost**\n❌ Not Available")
                
    else:
        st.error("❌ No models were loaded. Please check your `models` folder.")
#====================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Real-Time Prediction":
    st.header("🔮 Real-Time Text Analysis")
    st.markdown("Enter text below and select a model to get an instant prediction.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            model_choice = st.selectbox(
                "Choose a classifier:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x),
                key="realtime_model_selector"
            )
            #Test input or file upload option
            input_option = st.radio("Input method:", ("Paste Text", "Upload File"), horizontal=True, key="realtime_input_radio")
            text_input = ""

            if input_option == "Paste Text":
                text_input = st.text_area("Enter text here:", height=200, placeholder="Start typing...", key="realtime_textarea")
            else:
                uploaded_file = st.file_uploader("Upload .txt, .pdf, or .docx", type=['txt', 'pdf', 'docx'], key="realtime_file_upload")
                if uploaded_file:
                    with st.spinner("Reading file..."):
                        if uploaded_file.type == 'text/plain':
                            text_input = str(uploaded_file.read(), 'utf-8')
                        elif uploaded_file.type == 'application/pdf':
                            text_input = extract_text_from_pdf(uploaded_file)
                        else:
                            text_input = extract_text_from_docx(uploaded_file)

            if st.button("Analyze Text", type="primary", use_container_width=True, key="realtime_analyze_btn"):
                if text_input.strip():
                    with st.spinner(f'Analyzing with {model_choice}...'):
                        prediction, prob_dict = make_prediction(text_input, model_choice, models)

                        if prediction and prob_dict:
                            st.markdown("---")
                            st.markdown('<p class="prediction-header">Analysis Result</p>', unsafe_allow_html=True)

                            #Display prediction with an icon
                            icon = "🤖" if prediction == "AI-Generated" else "👤"
                            st.header(f"{icon} Prediction: **{prediction}**")

                            st.subheader("Confidence Score")
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("AI-Generated Probability", f"{prob_dict['AI-Generated']:.2%}")
                            with cols[1]:
                                st.metric("Human-Written Probability", f"{prob_dict['Human-Written']:.2%}")

                            # Bar chart for probabilities
                            prob_df = pd.DataFrame({
                                'Source': ['AI-Generated', 'Human-Written'],
                                'Probability': [prob_dict['AI-Generated'], prob_dict['Human-Written']]
                            })
                            st.bar_chart(prob_df.set_index('Source'))

                            # Explanation expander
                            with st.expander("Why did the model decide this?"):
                                st.markdown("""
                                **Disclaimer:** *This explanation is generalized. The model's decision is based on complex patterns learned from data.*

                                The model analyzes various linguistic features, including:
                                - **Text Complexity (Perplexity):** AI-generated text often has lower perplexity (i.e., it's more predictable).
                                - **Burstiness:** Human text often comes in 'bursts' of related words, while AI text can be more uniform.
                                - **Vocabulary Richness:** The variety and choice of words can differ between humans and AI.
                                - **Syntactic Structure:** Sentence length, structure, and consistency are key indicators.

                                Based on these features in your text, the model calculated the above probabilities.
                                """)

                            # Download report button
                            report_content = (
                                f"Analysis Report\n\nModel Used: {model_choice}\nPrediction: {prediction}\n\n"
                                f"Probabilities:\n- AI-Generated: {prob_dict['AI-Generated']:.2%}\n- Human-Written: {prob_dict['Human-Written']:.2%}\n\n"
                                f"--- Input Text ---\n{text_input}"
                            )
                            st.download_button("📥 Download Report", report_content, f"analysis_report.txt", "text/plain")
                        else:
                            st.error("Could not make a prediction. Please check the logs or try again.")
                else:
                    st.warning("Please enter some text to analyze.")
        else:
            st.error("No trained models are available for prediction.")
    else:
        st.warning("Models not loaded. Cannot perform prediction.")
# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a `.txt`, `.docx`, or `.pdf` file to classify its content.")
    
    if not models:
        st.info("Upload a file to get started.")
    else:
        available_models = get_available_models(models)

        if not available_models:
            st.error("No models available for batch processing.")
        else:
            # File upload widget
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['txt', 'pdf', 'docx'],
                help="Upload a document to determine if it was written by an AI or a human."
            )

            
        if uploaded_file:
                # Model selection dropdown
                model_choice = st.selectbox(
                    "Choose model for processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                if st.button("📊 Process Document", use_container_width=True):
                    try:
                        # Extract text from uploaded file
                        text_content = ""
                        if uploaded_file.type == "text/plain":
                            text_content = str(uploaded_file.read(), "utf-8")
                        elif uploaded_file.type == "application/pdf":
                            text_content = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text_content = extract_text_from_docx(uploaded_file)
                        
                        if not text_content.strip():
                            st.error("Could not extract any text from the document.")
                        else:
                            st.info(f"Successfully extracted {len(text_content.split())} words. Analyzing...")
                            
                            prediction, prob_dict = make_prediction(text_content, model_choice, models)
                            
                            if prediction and prob_dict:
                                st.markdown("---")
                                st.markdown('<p class="prediction-header">Document Analysis Result</p>', unsafe_allow_html=True)
                                
                                icon = "🤖" if prediction == "AI-Generated" else "👤"
                                st.header(f"{icon} The document is likely **{prediction}**")

                                st.subheader("Confidence Score")
                                cols = st.columns(2)
                                with cols[0]:
                                    st.metric("AI-Generated Probability", f"{prob_dict['AI-Generated']:.2%}")
                                with cols[1]:
                                    st.metric("Human-Written Probability", f"{prob_dict['Human-Written']:.2%}")
                                
                                # Prepare downloadable CSV report
                                report_data = {
                                    "Filename": [uploaded_file.name],
                                    "Model Used": [model_choice],
                                    "Prediction": [prediction],
                                    "AI_Prob": [prob_dict['AI-Generated']],
                                    "Human_Prob": [prob_dict['Human-Written']],
                                    "Word_Count": [len(text_content.split())],
                                    "Excerpt": [text_content[:500] + "..."]
                                }
                                report_df = pd.DataFrame(report_data)
                                st.download_button(
                                    label="📥 Download Analysis Report",
                                    data=report_df.to_csv(index=False).encode('utf-8'),
                                    file_name=f"analysis_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )

                    except Exception as e:
                        st.error(f"Error processing file: {e}")
# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Classifier Performance")
    st.markdown("Select a document and see how each model performs on the same input.")
    
    if models:
        available_models = get_available_models(models)
        
        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="The models will analyze this text...",
                height=150
            )
            
            if st.button("📊 Compare All Models", use_container_width=True) and comparison_text.strip():
                with st.spinner("Analyzing with all models..."):
                    st.subheader("🔍 Model Comparison Results")
                    
                    comparison_results = []
                    
                    for model_key, model_name in available_models:
                        prediction, prob_dict = make_prediction(comparison_text, model_key, models)
                        
                        if prediction and prob_dict:
                            comparison_results.append({
                                'Model': model_name.split(" ", 1)[1],  # Clean name without icon
                                'Prediction': prediction,
                                'AI Prob': f"{prob_dict['AI-Generated']:.2%}",
                                'Human Prob': f"{prob_dict['Human-Written']:.2%}"
                            })
                    
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        st.table(comparison_df)
                        
                        predictions = [r['Prediction'] for r in comparison_results]
                        if len(set(predictions)) == 1:
                            st.success(f"✅ All models agree: The text is likely **{predictions[0]}**")
                        else:
                            st.warning("⚠️ Models disagree on the prediction.")
                    else:
                        st.error("Failed to get predictions from models.")
        else:
            st.info("You need at least two models loaded to use the comparison feature.")
    else:
        st.warning("Models not loaded.")
# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model descriptions
        st.subheader("🔧 Available Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 Support Vector Machine (SVM)
            **Type:** Discriminative classifier  
            **Strength:** Effective in high-dimensional spaces, making it powerful for text data. It works by finding the optimal hyperplane that separates the two classes (AI vs. Human).
            """)
        with col2:
            st.markdown("""
            ### 🌳 Decision Tree
            **Type:** Tree-based supervised learning  
            **Strength:** Easy to understand and interpret. Uses decision rules to predict the target class based on feature values (e.g., word presence, n-grams).
            """)
        with col3:
            st.markdown("""
            ### 🚀 AdaBoost
            **Type:** Ensemble learning method  
            **Strength:** Combines multiple weak learners (like shallow decision trees) to build a strong classifier, improving accuracy while reducing overfitting.
            """)

        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        All models use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features. This technique reflects how important a word is to a document in a collection.

        **Parameters used:**
        - `max_features`: Between 1000–10000 depending on the model
        - `ngram_range`: (1, 2)
        - `stop_words`: English
        - `min_df`: 2, `max_df`: 0.95

        Each model uses its own fitted vectorizer inside the pipeline to ensure compatibility with its expected feature dimensions.
        """)

        # File status table
        st.subheader("📁 Model Files Status")
        file_status = []

        files_to_check = [
            ("Human_Vs_AI_Written_pipeline.pkl", "SVM Pipeline (Main)", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer (Backup)", models.get('vectorizer_available', False)),
            ("optimized_svm_model.pkl", "Standalone SVM Classifier", models.get('svm_available', False)),
            ("decision_tree_pipeline.pkl", "Decision Tree Pipeline", models.get('dt_available', False)),
            ("adaboost_pipeline.pkl", "AdaBoost Pipeline", models.get('ada_available', False))
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Optional training dataset note
        st.subheader("📚 Dataset Info")
        st.markdown("""
        - **Dataset:** AI vs Human Written Text Dataset  
        - **Labels:** `0 = Human-Written`, `1 = AI-Generated`  
        - **Preprocessing:** Custom cleaning, lemmatization, stopword removal  
        - **Split:** Models trained with stratified train-test split  
        - **Vectorization:** Model-specific TF-IDF pipelines
        """)
    
    else:
        st.warning("⚠️ Models not loaded. Please check the model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Real-Time Prediction"):
        st.write("""
        1. **Select a classifier** from the dropdown (SVM, Decision Tree, or AdaBoost).  
        2. **Enter text** in the text area or upload a file.  
        3. **Click 'Analyze Text'** to get classification result and confidence scores.
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:** Ensure you have a `.txt`, `.docx`, or `.pdf` document.  
        2. **Upload the file** using the uploader.  
        3. **Select a model** for processing.  
        4. **Click 'Process Document'** to analyze the entire file.  
        5. **Download a report** of the analysis if needed.
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to analyze.  
        2. **Click 'Compare All Models'** to get predictions from all available models.  
        3. **View comparison table** with side-by-side predictions and confidence scores.
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**

        - **Models not loading:**  
          - Ensure model `.pkl` files are in the `models/` directory.  
          - Required files:  
            - `Human_Vs_AI_Written_pipeline.pkl` (primary pipeline)  
            - `tfidf_vectorizer.pkl` (for individual models)  
            - `optimized_svm_model.pkl`  
            - `optimized_dt_model.pkl`  
            - `optimized_adaboost_model.pkl`  

        - **Prediction errors:**  
          - Make sure input text is not empty.  
          - Try shorter texts if memory errors occur.  
          - Ensure text contains readable characters.  

        - **File upload issues:**  
          - Supported formats: `.txt`, `.docx`, `.pdf`  
          - File encoding should be UTF-8.
        """)
    
    st.subheader("💻 Your Project Structure")
    st.code("""
ai_human_detection_project/
├── app.py
├── requirements.txt
├── models/
│   ├── Human_Vs_AI_Written_pipeline.pkl
│   ├── optimized_svm_model.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── adaboost_pipeline.pkl
│   ├── tfidf_vectorizer.pkl
├── data/
│   ├── training_data
│   └── test_data
├── sample_docs/
│   ├── AI Generated.pdf
│   ├── Human-Written.docx
│   └── AI.txt
├── notebooks/
│   └── Project_1.ipynb
├── README.md
    """, language='bash')
# ============================================================================
# FOOTER
# ============================================================================

#Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detection**
Built with Streamlit for the "AI vs Human Text Detection" project.

**Models:**
- 🤖 Support Vector Machine (SVM)
- 🌳 Decision Tree
- 🚀 AdaBoost
""")

#Mainpage footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | AI vs Human Text Detection App | Built by Samaya Niraula<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
</div>
""", unsafe_allow_html=True)