import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download NLTK stopwords silently
nltk.download('stopwords', quiet=True)

# Load pre-trained models with error handling
try:
    lr_classifier = joblib.load('lr_classifier.pkl')
    cv = joblib.load('cv_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files 'lr_classifier.pkl' or 'cv_vectorizer.pkl' not found! Please upload them.")
    st.stop()

# Initialize PorterStemmer
ps = PorterStemmer()

# Prediction function (unchanged from your notebook)
def stock_prediction(sample_news):
    sample_news = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_news)  # Remove non-letters
    sample_news = sample_news.lower()  # Convert to lowercase
    sample_news_words = sample_news.split()  # Split into words
    sample_news_words = [word for word in sample_news_words if word not in set(stopwords.words('english'))]  # Remove stopwords
    final_news = [ps.stem(word) for word in sample_news_words]  # Stem words
    final_news = ' '.join(final_news)  # Join back into a string
    temp = cv.transform([final_news]).toarray()  # Transform using CountVectorizer
    prediction = lr_classifier.predict(temp)[0]  # Predict
    return prediction

# Streamlit UI Setup
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Stock Prediction Tool")
    st.write("Predict stock price movements based on news headlines.")
    st.image("https://img.icons8.com/?size=100&id=111057&format=png&color=000000", caption="Stock Trends", width=80 ,)
    st.markdown("---")
# Main Content
st.title("📈 Stock Price Prediction")
st.markdown("#### Enter a News Headline to Predict Stock Movement")

# Input Section
with st.container():
    st.subheader("Your Headline")
    news_input = st.text_input("", placeholder="e.g., 'Tech company launches new product'", key="news_input")

# Prediction Button and Result
if st.button("🔍 Predict", type="primary", use_container_width=True):
    if news_input:
        with st.spinner("Analyzing headline..."):
            result = stock_prediction(news_input)
        # Display result with styling
        if result:
            st.markdown(
                '<div style="background-color:#33cc33;padding:10px;border-radius:5px;">'
                '📉 <b>Prediction:</b> Stock price will remain the same or go down</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background-color:#CC6666;padding:10px;border-radius:5px;">'
                '📈 <b>Prediction:</b> Stock price will go up</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter a headline to predict!")

# Footer
st.markdown("---")
st.write("Machine Learning Prediction")