import streamlit as st
import joblib
import os

st.title("🔍 Fake News Detector for Students")

# Load model and vectorizer
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

news = st.text_area("Paste News Article Here")

if st.button("Check News"):
    if news:
        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        words = news.split()
        summary = " ".join(words[:50])

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Verification Tips")
        st.write("✔ Verify the source")
        st.write("✔ Check other news websites")
        st.write("✔ Look for author credentials")
    else:
        st.warning("⚠️ Please paste a news article first!")

