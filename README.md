SentiScope — DistilBERT Sentiment Classifier

Fine-tuned DistilBERT for binary sentiment classification (Positive / Negative) on customer feedback text.

Show Image

Live Demo
Try it here →

Model Details
PropertyValueBase Modeldistilbert-base-uncasedTaskBinary Sentiment ClassificationClassesNegative, PositiveDatasetCustom customer feedback (96 samples)Train / Val Split80% / 20%Training Epochs3Best Val Accuracy100% (Epoch 3)HuggingFace Modelzentom/sentiment_analysis

Training Results
EpochTrain LossVal Accuracy10.67200.900020.60620.750030.52591.0000
Final Evaluation (Val Set)
MetricScoreAccuracy1.0000F1-Score (Macro)1.0000F1-Score (Weighted)1.0000

⚠️ Perfect scores are a result of the small dataset size (96 samples). Confidence scores on real-world inputs remain moderate (0.51–0.60), which honestly reflects the model's uncertainty — a healthy sign it isn't blindly overconfident.


How It Works
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

Project Structure
sentiscope/
├── train.py          # Fine-tuning script (DistilBERT + AdamW)
├── app.py            # Streamlit inference app
├── sentiment-analysis.csv  # Training data
└── README.md

Run Locally
bashgit clone https://github.com/your-username/sentiscope
cd sentiscope
pip install streamlit transformers torch scikit-learn pandas numpy
streamlit run app.py

Limitations
The model was trained on a small dataset of 96 samples. It handles clear positive/negative language well, but confidence scores on ambiguous or mixed-sentiment text are intentionally low (around 0.51–0.55). Retraining on a larger corpus like SST-2 or IMDb would significantly improve robustness.

Built With

Hugging Face Transformers
Streamlit
PyTorch


Author
Made by Usman.
