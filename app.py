import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import streamlit.components.v1 as components

nltk.download('stopwords', quiet=True)

# Load pre-trained models
try:
    lr_classifier = joblib.load('lr_classifier.pkl')
    cv = joblib.load('cv_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'lr_classifier.pkl' and 'cv_vectorizer.pkl' are present.")
    st.stop()

ps = PorterStemmer()

def stock_prediction(sample_news):
    sample_news = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_news)
    sample_news = sample_news.lower()
    sample_news_words = sample_news.split()
    sample_news_words = [word for word in sample_news_words if word not in set(stopwords.words('english'))]
    final_news = [ps.stem(word) for word in sample_news_words]
    final_news = ' '.join(final_news)
    temp = cv.transform([final_news]).toarray()
    prediction = lr_classifier.predict(temp)[0]
    return prediction

# Page Config
st.set_page_config(page_title="Sentiment Stocks", page_icon="📈", layout="wide")

# Custom CSS for styling and animations
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f4f7, #d9e4f5);
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #187bcd;
        color: white;
        transform: scale(1.05);
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Lottie animations
lottie_url_up = "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_url_down = "https://assets4.lottiefiles.com/packages/lf20_ydo1amjm.json"

# Sidebar
with st.sidebar:
    st.title("📊 Sentiment Stocks")
    st.write("Predict market movements based on news headlines.")
    st.markdown("---")
    st.info("Tips: 📝\n\n- Use clear financial headlines.\n- Avoid vague statements.\n- Industry-specific headlines improve accuracy.")

# Main Content (without dropdown)
st.title("📈 Stock Price Prediction Based on News")

with st.container():
    st.subheader("Enter a News Headline")
    news_input = st.text_input("", placeholder="e.g., Tech giant unveils breakthrough innovation")

if st.button("🔍 Predict", use_container_width=True):
    if news_input:
        with st.spinner("Analyzing the headline..."):
            result = stock_prediction(news_input)

        if result == 1:
            st.markdown(
                '<div class="result-box" style="background-color:#d4edda; color:#155724;">📈 <b>Prediction:</b> The stock price is likely to go up.</div>',
                unsafe_allow_html=True
            )
            components.html(f"""
                <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
                <lottie-player src="{lottie_url_up}" background="transparent" speed="1" style="width: 300px; height: 300px; display: block; margin: auto;" loop autoplay></lottie-player>
            """, height=300)

        else:
            st.markdown(
                '<div class="result-box" style="background-color:#f8d7da; color:#721c24;">📉 <b>Prediction:</b> The stock price may stay the same or go down.</div>',
                unsafe_allow_html=True
            )
            components.html(f"""
                <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
                <lottie-player src="{lottie_url_down}" background="transparent" speed="1" style="width: 300px; height: 300px; display: block; margin: auto;" loop autoplay></lottie-player>
            """, height=300)

    else:
        st.warning("⚠️ Please enter a headline to get a prediction.")

st.markdown("---")

# Footer
st.markdown("""
    <div style="text-align:center; margin-top:40px;">
        <p>🚀 Built with <b>Machine Learning Prediction Model</b></p>
        <p>Created by <a href="https://github.com/dhanushpavann" target="_blank">Dhanush Pavan</a></p>
    </div>
""", unsafe_allow_html=True)
