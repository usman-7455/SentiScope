import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from Hugging Face
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("zentom/sentiment_analysis")
    model = AutoModelForSequenceClassification.from_pretrained("zentom/sentiment_analysis")
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping (assuming binary: 0=Negative, 1=Positive)
labels = ["Negative", "Positive"]

# Preprocess function
def preprocess_text(text):
    text = text.lower().strip()
    return text

# Prediction function
def predict_sentiment(text):
    text = preprocess_text(text)
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        confidence = probs[pred_idx]
    return pred_label, confidence, probs

# Streamlit UI
st.title("Sentiment Analysis")
st.write("Enter text below to predict its sentiment (Positive or Negative).")

user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        pred_label, conf, probs = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{pred_label}**")
        st.write(f"Confidence: {conf:.3f}")
        st.write("Class Probabilities:")
        for i, cls in enumerate(labels):
            st.write(f"• {cls}: {probs[i]:.3f}")
