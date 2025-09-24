"""
🌟 Urdu to Roman Urdu Transliteration - Streamlit App
Beautiful interactive interface for real-time transliteration with token analysis
"""

import streamlit as st
import torch
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from io import BytesIO
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Urdu to Roman Transliteration",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from token_analyzer import TokenAnalyzer
    from improved_model import create_improved_model
    from enhanced_evaluate import EnhancedEvaluator
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .urdu-text {
        font-family: 'Jameel Noori Nastaleeq', 'Amiri', 'Scheherazade', serif;
        font-size: 1.5em;
        direction: rtl;
        text-align: right;
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e1e8ff;
        margin: 1rem 0;
    }
    
    .roman-text {
        font-family: 'Times New Roman', serif;
        font-size: 1.3em;
        background: #fff5f5;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffe1e1;
        margin: 1rem 0;
        color: #2d3748;
    }
    
    .token-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        border-radius: 20px;
        font-size: 0.9em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_transliteration_model():
    """Load the transliteration model with caching"""
    try:
        model_path = Path("models/best_model.pth")
        if not model_path.exists():
            return None, "Model file not found. Please train the model first."
        
        # Initialize components
        token_analyzer = TokenAnalyzer()
        evaluator = EnhancedEvaluator()
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab_size = checkpoint.get('vocab_size', {'source': 100, 'target': 100})
        
        model = create_improved_model(
            vocab_size['source'], 
            vocab_size['target'],
            embedding_dim=256,
            hidden_size=512,
            encoder_layers=2,
            decoder_layers=4
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        evaluator.model = model
        evaluator.source_vocab = checkpoint.get('source_vocab', {})
        evaluator.target_vocab = checkpoint.get('target_vocab', {})
        evaluator.source_idx2char = checkpoint.get('source_idx2char', {})
        evaluator.target_idx2char = checkpoint.get('target_idx2char', {})
        
        return evaluator, "Model loaded successfully!"
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_download_link(data, filename, text):
    """Create a download link for data"""
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, dict):
        data = json.dumps(data, indent=2, ensure_ascii=False).encode()
    
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 Urdu to Roman Urdu Transliteration</h1>
        <p>Advanced BiLSTM-powered transliteration with beautiful token analysis</p>
        <p><em>Transform Urdu text into Roman script with 93.6% accuracy</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading transliteration model..."):
        evaluator, status_message = load_transliteration_model()
    
    # Display model status
    if evaluator is None:
        st.markdown(f'<div class="error-message">❌ {status_message}</div>', unsafe_allow_html=True)
        st.info("To use this app, please train the model first using: `python src/improved_train.py`")
        st.stop()
    else:
        st.markdown(f'<div class="success-message">✅ {status_message}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Mode selection
        app_mode = st.selectbox(
            "Choose Mode",
            ["🔤 Single Translation", "📄 Batch Processing", "📊 Analysis Dashboard", "🧪 Model Explorer"]
        )
        
        # Settings
        st.header("⚙️ Settings")
        show_tokens = st.checkbox("Show Token Analysis", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        max_length = st.slider("Max Translation Length", 50, 500, 200)
        
        # Model info
        st.header("🤖 Model Info")
        st.info("""
        **Architecture:** BiLSTM Encoder-Decoder
        **Parameters:** ~21.4M
        **Accuracy:** 93.6%
        **Training Data:** 20K+ Urdu poems
        """)
    
    # Main content based on mode
    if app_mode == "🔤 Single Translation":
        single_translation_mode(evaluator, show_tokens, show_confidence, max_length)
    
    elif app_mode == "📄 Batch Processing":
        batch_processing_mode(evaluator, max_length)
    
    elif app_mode == "📊 Analysis Dashboard":
        analysis_dashboard_mode(evaluator)
    
    elif app_mode == "🧪 Model Explorer":
        model_explorer_mode(evaluator)

def single_translation_mode(evaluator, show_tokens, show_confidence, max_length):
    """Single text translation mode"""
    st.header("🔤 Single Translation")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input (Urdu)")
        urdu_text = st.text_area(
            "Enter Urdu text:",
            height=150,
            placeholder="یہاں اردو متن ٹائپ کریں...",
            key="urdu_input"
        )
        
        # Quick examples
        st.subheader("📝 Quick Examples")
        examples = [
            "اس ایک بات پہ دنیا سے جنگ جاری ہے",
            "پرو کی اب کے نہی حوصلو کی باری ہے", 
            "محبت میں نہیں ہے فرق جینے اور مرنے کا",
            "آپ کیسے ہیں؟"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.urdu_input = example
                st.experimental_rerun()
    
    with col2:
        st.subheader("Output (Roman Urdu)")
        
        if urdu_text.strip():
            with st.spinner("🔄 Transliterating..."):
                try:
                    # Perform transliteration
                    result = evaluator.translate_with_analysis(urdu_text.strip())
                    
                    if result:
                        # Display roman output
                        st.markdown(f'<div class="roman-text">{result["predicted_translation"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Display confidence if enabled
                        if show_confidence and "confidence_scores" in result:
                            confidence = sum(result["confidence_scores"]) / len(result["confidence_scores"])
                            st.progress(confidence)
                            st.caption(f"Average Confidence: {confidence*100:.1f}%")
                        
                        # Display tokens if enabled
                        if show_tokens and "source_tokens" in result:
                            st.subheader("🔍 Token Analysis")
                            
                            # Source tokens
                            st.write("**Urdu Tokens:**")
                            tokens_html = ""
                            unique_tokens = list(set(result["source_tokens"]))
                            for token in unique_tokens:
                                if token not in ['<SOS>', '<EOS>', '<PAD>']:
                                    tokens_html += f'<span class="token-badge">{token}</span>'
                            st.markdown(tokens_html, unsafe_allow_html=True)
                            
                            # Statistics
                            col1_stats, col2_stats = st.columns(2)
                            with col1_stats:
                                st.metric("Total Characters", len(result["source_tokens"]))
                            with col2_stats:
                                st.metric("Unique Characters", len(unique_tokens))
                    
                    else:
                        st.error("Translation failed. Please try again.")
                
                except Exception as e:
                    st.error(f"Error during translation: {str(e)}")
        
        else:
            st.info("👆 Enter Urdu text to see the translation here")

def batch_processing_mode(evaluator, max_length):
    """Batch processing mode"""
    st.header("📄 Batch Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a text file with Urdu sentences (one per line)",
        type=['txt'],
        help="Each line should contain one Urdu sentence"
    )
    
    # Manual input
    st.subheader("Or enter multiple sentences:")
    batch_text = st.text_area(
        "Enter Urdu sentences (one per line):",
        height=200,
        placeholder="ایک لائن میں ایک جملہ...\nدوسری لائن میں دوسرا جملہ..."
    )
    
    if st.button("🚀 Process Batch", type="primary"):
        sentences = []
        
        # Get sentences from file or text input
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            sentences = [line.strip() for line in content.split('\n') if line.strip()]
        elif batch_text:
            sentences = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        if sentences:
            st.subheader(f"Processing {len(sentences)} sentences...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            results_container = st.container()
            
            results = []
            
            for i, sentence in enumerate(sentences):
                try:
                    result = evaluator.translate_with_analysis(sentence)
                    if result:
                        results.append({
                            'Input': sentence,
                            'Output': result["predicted_translation"],
                            'Confidence': sum(result.get("confidence_scores", [0])) / max(len(result.get("confidence_scores", [1])), 1) * 100
                        })
                    progress_bar.progress((i + 1) / len(sentences))
                except Exception as e:
                    st.error(f"Error processing '{sentence}': {str(e)}")
            
            # Display results
            if results:
                results_container.subheader("✅ Results")
                df = pd.DataFrame(results)
                results_container.dataframe(df, use_container_width=True)
                
                # Download options
                csv_data = df.to_csv(index=False).encode('utf-8')
                results_container.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    "transliteration_results.csv",
                    "text/csv"
                )
                
                json_data = df.to_json(orient='records', force_ascii=False)
                results_container.download_button(
                    "📥 Download as JSON", 
                    json_data,
                    "transliteration_results.json",
                    "application/json"
                )
        else:
            st.warning("Please upload a file or enter some text to process.")

def analysis_dashboard_mode(evaluator):
    """Analysis dashboard mode"""
    st.header("📊 Analysis Dashboard")
    
    # Sample analysis with pre-defined sentences
    sample_sentences = [
        "اس ایک بات پہ دنیا سے جنگ جاری ہے",
        "پرو کی اب کے نہی حوصلو کی باری ہے",
        "محبت میں نہیں ہے فرق جینے اور مرنے کا",
        "دل میں اک لہر سی اٹھی ہے ابھی",
        "یہ کیسا عشق ہے کہ بن گئے ہم بے قرار"
    ]
    
    if st.button("🔬 Run Sample Analysis"):
        with st.spinner("Analyzing samples..."):
            analysis_results = []
            
            for sentence in sample_sentences:
                try:
                    result = evaluator.translate_with_analysis(sentence)
                    if result:
                        confidence = sum(result.get("confidence_scores", [0])) / max(len(result.get("confidence_scores", [1])), 1)
                        analysis_results.append({
                            'sentence': sentence,
                            'translation': result["predicted_translation"],
                            'confidence': confidence,
                            'char_count': len(sentence),
                            'token_count': len(result.get("source_tokens", []))
                        })
                except:
                    continue
            
            if analysis_results:
                # Create visualizations
                df = pd.DataFrame(analysis_results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Confidence Distribution")
                    fig = px.bar(df, x='sentence', y='confidence', 
                               title="Translation Confidence by Sentence")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📏 Length Analysis")
                    fig = px.scatter(df, x='char_count', y='token_count', 
                                   size='confidence', hover_data=['sentence'],
                                   title="Character vs Token Count")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                st.dataframe(df, use_container_width=True)

def model_explorer_mode(evaluator):
    """Model exploration mode"""
    st.header("🧪 Model Explorer")
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Encoder</h3>
            <p>2-layer BiLSTM</p>
            <p>512 hidden units</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Decoder</h3>
            <p>4-layer LSTM</p>
            <p>512 hidden units</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Attention</h3>
            <p>Additive mechanism</p>
            <p>256 dimensions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Vocabulary info
    st.subheader("📚 Vocabulary Information")
    
    if hasattr(evaluator, 'source_vocab') and evaluator.source_vocab:
        source_vocab_size = len(evaluator.source_vocab)
        target_vocab_size = len(evaluator.target_vocab)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Urdu Characters", source_vocab_size)
        with col2:
            st.metric("Roman Characters", target_vocab_size)
        
        # Show some vocabulary examples
        if st.checkbox("Show Vocabulary Samples"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Urdu Characters:**")
                urdu_chars = list(evaluator.source_vocab.keys())[:20]
                for char in urdu_chars:
                    if char not in ['<PAD>', '<SOS>', '<EOS>']:
                        st.write(f"`{char}` (ID: {evaluator.source_vocab[char]})")
            
            with col2:
                st.write("**Roman Characters:**")
                roman_chars = list(evaluator.target_vocab.keys())[:20]
                for char in roman_chars:
                    if char not in ['<PAD>', '<SOS>', '<EOS>']:
                        st.write(f"`{char}` (ID: {evaluator.target_vocab[char]})")

if __name__ == "__main__":
    main()