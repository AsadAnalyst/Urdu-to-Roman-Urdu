# 🌟 Streamlit Deployment Guide

## Deploy to Streamlit Community Cloud

### 1. 📋 Prerequisites
- GitHub repository (public)
- Streamlit account
- Model files uploaded separately (due to size limits)

### 2. 🚀 Quick Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repo: `AsadAnalyst/Urdu-To-Roman-Urdu`
4. Set main file: `streamlit_app.py`
5. Click "Deploy"

### 3. 🔧 Configuration

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "serif"

[server]
maxUploadSize = 200
maxMessageSize = 200
```

### 4. 📦 Model Handling

Since model files are large, users need to:

**Option A: Download Pre-trained Models**
```python
# In streamlit_app.py, add automatic model download
import urllib.request
import os

@st.cache_data
def download_model():
    model_url = "YOUR_MODEL_DOWNLOAD_URL"  # Google Drive/Hugging Face
    if not os.path.exists("models/best_model.pth"):
        st.info("Downloading model... (first time only)")
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(model_url, "models/best_model.pth")
    return True
```

**Option B: Use Hugging Face Hub**
```python
# Install: pip install huggingface-hub
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model_from_hf():
    model_path = hf_hub_download(
        repo_id="your-username/urdu-roman-model",
        filename="best_model.pth"
    )
    return torch.load(model_path, map_location='cpu')
```

### 5. 🌐 Custom Domain (Optional)

Add to your repo's `.streamlit/config.toml`:
```toml
[server]
baseUrlPath = "/urdu-transliteration"
```

### 6. 📊 Environment Variables

For secrets (API keys, etc.), use Streamlit Cloud secrets:
1. Go to app settings
2. Add secrets in TOML format:
```toml
[general]
api_key = "your-api-key"
```

### 7. 🔄 Auto-deployment

- Pushes to main branch auto-deploy
- View logs in Streamlit Cloud dashboard
- Monitor app health and usage

### 8. 📈 Production Tips

```python
# Add error handling
try:
    result = model.predict(input_text)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.info("Please try again or contact support")

# Add caching for better performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_prediction(text):
    return model.predict(text)

# Add usage analytics
import streamlit.components.v1 as components
components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
""", height=0)
```

---

## 🎯 Local Development

### Run Locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Debug Mode:
```bash
streamlit run streamlit_app.py --logger.level debug
```

### Custom Port:
```bash
streamlit run streamlit_app.py --server.port 8080
```

---

## 📱 Mobile Optimization

The app is already mobile-responsive with:
- ✅ Responsive layout
- ✅ Touch-friendly buttons  
- ✅ Mobile-optimized text areas
- ✅ Swipe-friendly UI components

---

## 🚀 Your App URL
After deployment: `https://your-app-name.streamlit.app`

Share this link to let the world use your amazing Urdu transliteration system! 🌟