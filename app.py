import streamlit as st
from pathlib import Path
from src.predict import SentimentPredictor

st.set_page_config(
    page_title="IMDb Sentiment Intelligence",
    page_icon="🎬",
    layout="centered",
)

MODEL_DIR = Path(__file__).resolve().parent / "models" / "distilbert-imdb-sentiment"

st.title("🎬 IMDb Sentiment Intelligence")
st.caption("DistilBERT fine-tuned on the provided IMDb review dataset")

@st.cache_resource
def load_predictor():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the Jupyter notebook first."
        )
    return SentimentPredictor(MODEL_DIR)

try:
    predictor = load_predictor()
except Exception as exc:
    st.warning(str(exc))
    st.stop()

review = st.text_area(
    "Enter a movie review",
    height=180,
    placeholder="Example: The story was engaging and the performances were excellent...",
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.info("Please enter a review.")
    else:
        result = predictor.predict(review.strip())
        label = result["label"]
        confidence = result["confidence"]

        if label == "POSITIVE":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric("Model confidence", f"{confidence:.2%}")

        st.subheader("Class probabilities")
        for name, probability in result["probabilities"].items():
            st.progress(probability, text=f"{name}: {probability:.2%}")
