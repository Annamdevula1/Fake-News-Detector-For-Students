import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
 st.image("FAKE NEWS IMAGE.jpeg", width= 1000)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f4f8fb;
}

.title{
    text-align:center;
    color:#0F4C81;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555555;
    font-size:18px;
    margin-bottom:25px;
}

.stTextArea textarea{
    border-radius:10px;
    border:2px solid #0F4C81;
    font-size:16px;
}

.stButton>button{
    width:100%;
    background:#0F4C81;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1565C0;
    color:white;
}

.result-box{
    padding:15px;
    border-radius:10px;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)
# =====================================================
# Header
# =====================================================

st.markdown(
    "<h1 class='title'>📰 Fake News Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Detect whether a news article is <b>Real</b> or <b>Fake</b> using Machine Learning (Logistic Regression + TF-IDF).</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/480/news.png",
        width=120
    )

    st.title("📋 Project Information")

    st.info("""
**Project Name**
Fake News Detection System
""")

    st.success("""
**Machine Learning Model**
Logistic Regression
""")

    st.success("""
**Text Vectorizer**
TF-IDF Vectorizer
""")

    st.info("""
**Dataset Details**

• Total Records: 45,757

• Text Column: text

• Target Column: label

• Fake News = 0

• Real News = 1
""")



    



    st.subheader("ℹ About")

    st.write("""
This application predicts whether a news article is **Real** or **Fake** using a Machine Learning model trained on a labeled news dataset.
""")

st.markdown("### ✍ Enter News Article")

news_text = st.text_area(
    "",
    height=250,
    placeholder="Paste or type a news article here..."
)
# =====================================================
# Prediction Section
# =====================================================

predict = st.button("🔍 Predict News")

if predict:

    if news_text.strip() == "":
        st.warning("⚠ Please enter a news article before clicking Predict.")

    else:

        # Transform text
        transformed_text = vectorizer.transform([news_text])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Prediction Probability
        probability = model.predict_proba(transformed_text)[0]
        confidence = max(probability) * 100

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        # -----------------------------
        # REAL NEWS
        # -----------------------------
        if prediction == 1:

            st.success("✅ This News is **REAL**")

            st.progress(int(confidence))

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.balloons()

        # -----------------------------
        # FAKE NEWS
        # -----------------------------
        else:

            st.error("❌ This News is **FAKE**")

            st.progress(int(confidence))

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

        st.markdown("---")

        st.subheader("📈 Prediction Probabilities")

        st.write(f"🟢 **Real News Probability:** {probability[1]*100:.2f}%")

        st.write(f"🔴 **Fake News Probability:** {probability[0]*100:.2f}%")

        chart_data = {
            "Real News": probability[1],
            "Fake News": probability[0]
        }

        st.bar_chart(chart_data)

        st.markdown("---")

        if prediction == 1:
            st.info("""

# =====================================================
# Project Information
# =====================================================

st.markdown("---")

st.header("📌 Project Overview")

st.write("""
The Fake News Detector for studentsis a Machine Learning application
that classifies or predicts whether a news article is Real or Fake Using NLP Techniques.

It provides fast and accurate predictions through an easy to use streamlit interface.
""")

# =====================================================
# Model Information
# =====================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 🤖 Model
Logistic Regression
""")

with col2:
    st.info("""
### 📄 Dataset
45,757 News Articles
""")

with col3:
    st.info("""
### 🔤 Features
TF-IDF Vectorizer
""")

# =====================================================
# Tips
# =====================================================

st.markdown("---")

st.subheader("💡 Tips for Better Prediction")

st.write("""
- Enter a complete news article instead of only a headline.
- Longer text generally produces more reliable predictions.
- The prediction is based only on the text entered.
- Always verify important news from trusted official sources.
""")

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>

### 📰 Fake News Detection System

Developed using ❤️ with **Python, Streamlit, Scikit-learn & Machine Learning**

**Model:** Logistic Regression

**Text Processing:** TF-IDF Vectorizer

© 2026 All Rights Reserved

</div>
""",
unsafe_allow_html=True
)
# =====================================================
# Extra Features
# =====================================================

st.markdown("---")
st.header("⚙️ Extra Features")

# -----------------------------
# Dark Mode Toggle
# -----------------------------
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp{
        background-color:#121212;
        color:white;
    }

    h1,h2,h3,h4,h5,h6,p,label,span{
        color:white !important;
    }

    .stTextArea textarea{
        background:#1e1e1e;
        color:white;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Sample News
# -----------------------------
st.subheader("📰 Sample News")

sample_news = """
The government announced a new education policy to improve digital learning
across the country. The initiative includes infrastructure development,
teacher training, and better internet connectivity in rural areas.
"""

if st.button("📄 Load Sample News"):
    st.text_area(
        "Sample News",
        value=sample_news,
        height=200,
        disabled=True
    )

# -----------------------------
# Prediction History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if predict and news_text.strip() != "":
    result = "Real News" if prediction == 1 else "Fake News"

    st.session_state.history.append({
        "Prediction": result,
        "Confidence": f"{confidence:.2f}%"
    })

st.markdown("---")

st.subheader("📜 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions made yet.")
else:
    st.table(st.session_state.history)

# -----------------------------
# Download Result
# -----------------------------
if predict and news_text.strip() != "":

    report = f"""
Fake News Detection Report

Prediction :
{"Real News" if prediction == 1 else "Fake News"}

Confidence :
{confidence:.2f}%

Real Probability :
{probability[1]*100:.2f}%

Fake Probability :
{probability[0]*100:.2f}%

News Text :
{news_text}
"""

    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="prediction_report.txt",
        mime="text/plain"
    )

# -----------------------------
# Clear Input
# -----------------------------
if st.button("🗑 Clear Screen"):
    st.rerun()

# -----------------------------
# Thank You Message
# -----------------------------
st.markdown("---")

st.success("""
🎉 Thank you for using the Fake News Detection System!

We hope this application helps you identify potentially fake news quickly and effectively.
""")
