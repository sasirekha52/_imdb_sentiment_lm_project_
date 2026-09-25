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
        
    try:
        # 1. Attempt standard loading
        return SentimentPredictor(MODEL_DIR)
    except Exception as e:
        # 2. Check if the error is related to sentencepiece / tokenizer backend
        if "tokenizer" in str(e).lower() or "sentencepiece" in str(e).lower():
            import transformers
            from transformers import AutoTokenizer, pipeline
            
            # Monkey-patch or override pipeline creation to bypass the missing binary package
            st.info("Configuring native backend tokenizer fallback...")
            
            # Explicitly instantiate the slow tokenizer variant
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
            
            # Reconstruct the pipeline manually without relying on fast rust tokenizers
            fallback_pipeline = pipeline(
                "sentiment-analysis", 
                model=str(MODEL_DIR), 
                tokenizer=tokenizer
            )
            
            # Initialize predictor wrapper
            predictor_instance = SentimentPredictor(MODEL_DIR)
            
            # If your custom SentimentPredictor exposes the pipeline object, override it:
            if hasattr(predictor_instance, 'pipeline'):
                predictor_instance.pipeline = fallback_pipeline
            elif hasattr(predictor_instance, 'clf'):
                predictor_instance.clf = fallback_pipeline
                
            return predictor_instance
        else:
            # Re-raise if it's an unrelated exception
            raise e

try:
    predictor = load_predictor()
except Exception as exc:
    st.error(f"Failed to initialize model: {str(exc)}")
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
