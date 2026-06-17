import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load(open("fake_news_model.pkl", "rb"))
vectorizer = joblib.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector for Students")

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
        st.write("""
        ✔ Verify the source

        ✔ Check publication date

        ✔ Cross-check with trusted websites

        ✔ Look for evidence and references
        """)
