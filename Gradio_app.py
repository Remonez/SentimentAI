import gradio as gr
import pandas as pd
import joblib
import os

# Load the model and vectorizer
try:
    rf_model = joblib.load('models/rf_model_sqrt.joblib')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
except Exception as e:
    raise RuntimeError(f"Failed to load the model or vectorizer: {e}")

# Label mapping for predictions
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to predict sentiment for a single input
def predict_sentiment(feedback):
    try:
        input_vectorized = tfidf_vectorizer.transform([feedback])
        prediction = rf_model.predict(input_vectorized)[0]
        sentiment = label_mapping.get(prediction, "Unknown")
        return f"The predicted sentiment is: {sentiment}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# Function to predict sentiments for CSV and provide direct download link
def predict_from_csv(file):
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(file.name, on_bad_lines="skip")
        
        # Ensure the required column exists
        if "feedback" not in data.columns:
            return "❗ The uploaded CSV file must contain a column named 'feedback'.", None
        
        # Process the feedback column
        feedback_texts = data["feedback"].fillna("").tolist()
        input_vectorized = tfidf_vectorizer.transform(feedback_texts)
        predictions = rf_model.predict(input_vectorized)
        
        # Map predictions to sentiments
        sentiments = [label_mapping.get(pred, "Unknown") for pred in predictions]
        data["Sentiment"] = sentiments
        
        # Save the updated data to a file in the app directory
        file_path = os.path.join(os.getcwd(), "predictions_with_sentiments.csv")
        data.to_csv(file_path, index=False)
        
        # Generate a Markdown link for direct file download
        download_link = f"[⬇️ Click here to download the results CSV](file://{file_path})"
        return data, gr.Markdown(download_link)
    except Exception as e:
        return f"⚠️ An error occurred while processing the file: {e}", None

# Gradio interface for single feedback input
single_feedback_interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter customer feedback...",
        label="📝 Enter Feedback"
    ),
    outputs=gr.Text(label="Sentiment Prediction"),
    title="Interactive Customer Feedback Sentiment Predictor",
    description="Analyze customer feedback to predict whether it is positive, neutral, or negative.",
    flagging_mode="never"
)

# Gradio interface for CSV input
csv_file_interface = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(
        label="📂 Upload CSV File",
        file_types=[".csv"]
    ),
    outputs=[
        gr.Dataframe(label="📊 Results with Predicted Sentiments"),
        gr.Markdown(label="⬇️ Download Predictions (CSV)")
    ],
    title="CSV Sentiment Analysis",
    description="Upload a CSV file containing a **'feedback'** column to analyze sentiments for multiple entries.",
    flagging_mode="never" 
)

# Combine both interfaces in a Tabbed layout
app = gr.TabbedInterface(
    interface_list=[
        single_feedback_interface,
        csv_file_interface
    ],
    tab_names=["Single Feedback Prediction", "CSV File Analysis"]
)

# Run the app
if __name__ == "__main__":
    app.launch()
