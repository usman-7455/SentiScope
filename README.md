# SentiScope — DistilBERT Sentiment Classifier

> Fine-tuned DistilBERT for binary sentiment classification (Positive / Negative) on customer feedback text.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentimentanalysispredict-jef7ngrbqscm6q5mlgm4ad.streamlit.app/)

---

## Live Demo

**[Try it here →](https://sentimentanalysispredict-jef7ngrbqscm6q5mlgm4ad.streamlit.app/)**

---

## Model Details

| Property | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Task | Binary Sentiment Classification |
| Classes | Negative, Positive |
| Dataset | Custom customer feedback (96 samples) |
| Train / Val Split | 80% / 20% |
| Training Epochs | 3 |
| Best Val Accuracy | 100% (Epoch 3) |
| HuggingFace Model | [`zentom/sentiment_analysis`](https://huggingface.co/zentom/sentiment_analysis) |

---

## Training Results

| Epoch | Train Loss | Val Accuracy |
|---|---|---|
| 1 | 0.6720 | 0.9000 |
| 2 | 0.6062 | 0.7500 |
| 3 | 0.5259 | **1.0000** |

**Final Evaluation (Val Set)**

| Metric | Score |
|---|---|
| Accuracy | 1.0000 |
| F1-Score (Macro) | 1.0000 |
| F1-Score (Weighted) | 1.0000 |

> ⚠️ Perfect scores are a result of the small dataset size (96 samples). Confidence scores on real-world inputs remain moderate (0.51–0.60), which honestly reflects the model's uncertainty — a healthy sign it isn't blindly overconfident.

---

## How It Works

```
Raw Text
   ↓
Lowercase + whitespace normalization
   ↓
DistilBERT Tokenizer (max_length=128, padded)
   ↓
DistilBertForSequenceClassification
   ↓
Softmax → [P(Negative), P(Positive)]
   ↓
argmax → Predicted Label + Confidence
```

---

## Project Structure

```
sentiscope/
├── train.py          # Fine-tuning script (DistilBERT + AdamW)
├── app.py            # Streamlit inference app
├── sentiment-analysis.csv  # Training data
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/your-username/sentiscope
cd sentiscope
pip install streamlit transformers torch scikit-learn pandas numpy
streamlit run app.py
```

---

## Limitations

The model was trained on a small dataset of 96 samples. It handles clear positive/negative language well, but confidence scores on ambiguous or mixed-sentiment text are intentionally low (around 0.51–0.55). Retraining on a larger corpus like SST-2 or IMDb would significantly improve robustness.

---

## Built With

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)

---

## Author

Made by **Usman**.
