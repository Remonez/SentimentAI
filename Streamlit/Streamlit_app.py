import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
try:
    rf_model = joblib.load('models/rf_model_sqrt.joblib')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
except Exception as e:
    st.error(f"⚠️ Failed to load the model or vectorizer: {e}")
    st.stop()

# Label mapping for sentiment prediction
label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Streamlit app setup
st.set_page_config(page_title="Customer Feedback Sentiment Analysis", page_icon="🔍", layout="wide")

# Title and description
st.title("Customer Feedback Sentiment Analysis")
st.markdown("""
    Enter individual customer feedback or upload a CSV file to analyze sentiments: **Negative**, **Neutral**, or **Positive**.
""", unsafe_allow_html=True)

# Tab layout for user input options
tab1, tab2 = st.tabs(["Single Feedback", "Upload CSV File"])

# **Single Feedback Input**
with tab1:
    user_input = st.text_area(
        "📝 **Enter customer feedback**:",
        height=200,
        max_chars=1000,
        placeholder="E.g., 'The product quality is great, but delivery was delayed.'"
    )

    if st.button("🔮 Predict Sentiment", key="single"):
        if user_input.strip():
            try:
                # Transform and predict
                input_vectorized = tfidf_vectorizer.transform([user_input])
                prediction = rf_model.predict(input_vectorized)[0]
                sentiment = label_mapping.get(prediction, "Unknown")

                # Display sentiment with colors
                sentiment_colors = {
                    "Negative": "red",
                    "Neutral": "orange",
                    "Positive": "green"
                }
                st.markdown(f"### **Predicted Sentiment**: <span style='color:{sentiment_colors[sentiment]}'>**{sentiment}**</span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ An error occurred during prediction: {e}")
        else:
            st.warning("❗ Please enter valid feedback text to predict sentiment.")

# **CSV File Input**
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV file with customer feedback. **The file must include a column named 'feedback'**", type="csv")

    if uploaded_file:
        try:
            # Read CSV and display preview
            data = pd.read_csv(uploaded_file, on_bad_lines='skip')
            if "feedback" in data.columns:  # Check for required column
                st.write("Preview of uploaded data:")
                st.write(data.head())

                if st.button("🔮 Predict Sentiments", key="csv"):
                    # Process the 'feedback' column
                    feedback_texts = data["feedback"].fillna("").tolist()
                    input_vectorized = tfidf_vectorizer.transform(feedback_texts)
                    predictions = rf_model.predict(input_vectorized)

                    # Map predictions to sentiments
                    sentiments = [label_mapping.get(pred, "Unknown") for pred in predictions]
                    data["Sentiment"] = sentiments

                    st.write("📊 Results with Predicted Sentiments:")
                    st.write(data)

                    # Allow user to download the results
                    csv_data = data.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions", csv_data, "predicted_sentiments.csv", "text/csv")
            else:
                st.warning("❗ The uploaded CSV file must contain a column named 'feedback'.")
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the file: {e}")

# Footer
st.markdown("""
    ---
    🧑‍💻 **Developed by Remon Ez**     📧 [Email](mailto:ezremon88@gmail.com)   |   🐙 [GitHub](https://github.com/Remonez)
""", unsafe_allow_html=True)
