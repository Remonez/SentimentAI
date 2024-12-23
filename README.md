<div align="center">
  <h1 style="font-size: 45px;"><b><i>SentimentAI</i></b></h1>
</div>


# ***Interactive Customer Feedback Sentiment Predictor***

## **Overview**
This project predicts the sentiment (Positive, Neutral, Negative) of customer feedback using machine learning. It provides:
- An **interactive interface** for single feedback inputs or bulk CSV file uploads.
- **TF-IDF embeddings** combined with a **Random Forest model** for high-accuracy predictions.

---

## **Problem Statement**
Understanding customer sentiment is vital for businesses to improve their products and services. This project addresses the challenge by analyzing customer feedback to provide actionable insights into sentiment trends.

---

## **Technologies Used**
- **Machine Learning**: Scikit-learn (Random Forest)
- **NLP**: TF-IDF Vectorization
- **Deployment**: Gradio, Docker

---

## **Key Features**
1. **Interactive Sentiment Prediction**:
   - Single feedback input for real-time sentiment analysis.
   - Bulk prediction using a CSV file upload.

2. **Robust Model**:
   - Random Forest combined with TF-IDF achieved **90% accuracy**.
   - Handles large datasets with optimized predictions.

3. **Dockerized Deployment**:
   - Run the application in a lightweight and portable container using Docker.

---

## **How to Run with Docker**

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Remonez/SentimentAI.git
   cd SentimentAI
   ```

2. Build the Docker image:
   ```bash
   docker build -t sentiment-predictor .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 7860:7860 sentiment-predictor
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

## **Future Improvements**
- Fine-tune Transformer models (e.g., DistilBERT).
- Enhance multilingual support for sentiment analysis.
- Add live data scraping for dynamic feedback analysis.
