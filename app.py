"""
🌟 Beautiful Urdu to Roman Transliteration Web App
Flask Backend Server with Token Analysis and Attention Visualization
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from token_analyzer import TokenAnalyzer
from improved_model import create_improved_model
from enhanced_evaluate import EnhancedEvaluator

app = Flask(__name__)
CORS(app)

class TransliterationAPI:
    """API class for handling transliteration requests"""
    
    def __init__(self):
        """Initialize the transliteration model and analyzer"""
        self.model = None
        self.token_analyzer = None
        self.evaluator = None
        self.is_loaded = False
        
        # Try to load model on startup
        self.load_model()
    
    def load_model(self):
        """Load the trained model and initialize components"""
        try:
            model_path = 'models/best_model.pth'
            config_path = 'improved_config.json'
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found at {model_path}")
                return False
            
            if not os.path.exists(config_path):
                print(f"❌ Config not found at {config_path}")
                return False
            
            # Initialize evaluator which loads everything
            self.evaluator = EnhancedEvaluator(model_path, config_path)
            self.model = self.evaluator.model
            self.token_analyzer = self.evaluator.token_analyzer
            
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def transliterate(self, urdu_text):
        """Transliterate Urdu text to Roman"""
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Model not loaded. Please check if model files exist.'
            }
        
        try:
            # Get detailed analysis
            analysis = self.token_analyzer.analyze_sample(urdu_text)
            
            return {
                'success': True,
                'urdu_text': urdu_text,
                'roman_text': analysis['prediction']['text'],
                'source_tokens': analysis['source']['tokens'],
                'predicted_tokens': analysis['prediction']['tokens'],
                'source_token_ids': analysis['source']['token_ids'],
                'predicted_token_ids': analysis['prediction']['token_ids'],
                'confidence_scores': analysis['prediction']['confidence_scores'],
                'alignments': analysis['alignments'][:10] if analysis['alignments'] else []  # Top 10 alignments
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Transliteration failed: {str(e)}'
            }
    
    def create_attention_heatmap(self, analysis_data):
        """Create attention heatmap and return as base64 encoded image - DISABLED"""
        return None

# Initialize API
api = TransliterationAPI()

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/api/transliterate', methods=['POST'])
def transliterate_text():
    """API endpoint for transliteration"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'success': False,
            'error': 'No text provided'
        }), 400
    
    urdu_text = data['text'].strip()
    
    if not urdu_text:
        return jsonify({
            'success': False,
            'error': 'Empty text provided'
        }), 400
    
    # Perform transliteration
    result = api.transliterate(urdu_text)
    
    return jsonify(result)

@app.route('/api/attention_heatmap', methods=['POST'])
def get_attention_heatmap():
    """API endpoint for attention heatmap - DISABLED"""
    return jsonify({
        'success': False,
        'error': 'Attention visualization has been disabled'
    }), 404

@app.route('/api/model_status')
def model_status():
    """Check if model is loaded"""
    return jsonify({
        'loaded': api.is_loaded,
        'message': 'Model is ready' if api.is_loaded else 'Model not loaded'
    })

@app.route('/api/reload_model', methods=['POST'])
def reload_model():
    """Reload the model"""
    success = api.load_model()
    return jsonify({
        'success': success,
        'message': 'Model reloaded successfully' if success else 'Failed to reload model'
    })

if __name__ == '__main__':
    print("🚀 Starting Beautiful Urdu to Roman Transliteration Web App...")
    print("="*60)
    print("🌟 Features:")
    print("   ✨ Real-time transliteration")
    print("   🔍 Token analysis and visualization")
    print("   📱 Responsive beautiful interface")
    print("="*60)
    
    if api.is_loaded:
        print("✅ Model loaded successfully!")
        print(f"   📊 Model parameters: {api.evaluator._count_parameters():,}")
        print(f"   🔤 Source vocabulary: {len(api.evaluator.src_vocab)} characters")
        print(f"   🔤 Target vocabulary: {len(api.evaluator.tgt_vocab)} characters")
    else:
        print("⚠️  Model not loaded. Some features may not work.")
    
    print("\\n🌐 Starting server at: http://localhost:5000")
    print("💡 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)