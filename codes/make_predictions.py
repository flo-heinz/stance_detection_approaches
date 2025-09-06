"""
This script was generated with the assistance of ChatGPT based on the following
kinds of prompts/instructions:

1. General request / purpose
   - "Build a regression model using SciBERT (via HuggingFace Transformers and TensorFlow/Keras)
      to predict stance scores from research article abstracts and titles."

2. Data handling
   - "Load input data from a JSON file (titles, abstracts, stance labels) and
      output predictions with stance scores and categories into another JSON file."

3. Model architecture
   - "Use SciBERT as encoder, take its [CLS]/pooled output, and add a regression head
      with tanh activation to produce a score in [-1.0, 1.0]."

4. Training and loading
   - "If pretrained weights exist, load them; otherwise train on the given dataset
      using Mean Squared Error loss."

5. Prediction and output format
   - "Convert raw stance_score into categorical labels ('in favor', 'neutral', 'against')
      and save results with titles, abstracts, gold labels, and predictions."

6. Developer experience
   - "Show clear console messages (with emojis if desired), keep code robust
      against missing token_type_ids and numpy serialization issues."
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.data import Dataset

# configuration
MODEL_DIR     = r'C:\Users\flori\OneDrive\Desktop\Programmier Pro\TrainModel\2025-09-02_21-43-47'
WEIGHTS_FILE  = 'scibert_regression.weights.h5'
CODES_DIR = Path(__file__).resolve().parent
DATA_DIR  = CODES_DIR.parent / "data"
DATA_FILE = DATA_DIR / "evaluation_part.json"
OUTPUT_FILE = DATA_DIR / "NLP-Predictions_scibert_regression.json"



MAX_LEN       = 300        # max tokens per sequence
BATCH_SIZE    = 16
EPOCHS        = 5
LEARNING_RATE = 5e-5       # typical fine-tuning LR

# load dataset
print("📂 Loading dataset...")
df = pd.read_json(DATA_FILE)

# combine title and abstract with a [SEP] token so BERT can attend to both
texts = df.apply(lambda row: f"{row['title']} [SEP] {row['abstract']}", axis=1).tolist()

print("🔤 Tokenizing texts...")
# use local tokenizer if available, otherwise fallback to SciBERT from HF hub
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
except Exception:
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# tokenize text into input_ids, attention_mask, (and possibly) token_type_ids
encodings = tokenizer(
    texts,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

# ensure token_type_ids exist (some BERT variants omit them)
if 'token_type_ids' not in encodings:
    encodings['token_type_ids'] = tf.zeros_like(encodings['input_ids'])

encoded_dataset = Dataset.from_tensor_slices(dict(encodings)).batch(BATCH_SIZE)

# build regression model around SciBERT
print("🧠 Building regression model...")

input_ids      = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
token_type_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="token_type_ids")

# load SciBERT (huggingface model, weights converted from PyTorch)
bert = TFBertModel.from_pretrained("allenai/scibert_scivocab_uncased", from_pt=True)

# wrap BERT forward pass in a Lambda layer to integrate into Keras model
def bert_call(tensors):
    ids, mask, types = tensors
    outputs = bert(input_ids=ids, attention_mask=mask, token_type_ids=types)
    return outputs.pooler_output  # use CLS pooled embedding (batch, 768)

pooled_output = Lambda(
    bert_call,
    name="bert_pooled_output",
    output_shape=(768,),
    dtype='float32'
)([input_ids, attention_mask, token_type_ids])

# regression head: one neuron with tanh to produce scores in [-1, 1]
stance_output = Dense(1, activation='tanh', name="stance_score")(pooled_output)

# assemble model
model = Model(
    inputs={"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
    outputs=stance_output
)

# compile with Adam optimizer and MSE loss
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=MeanSquaredError()
)

# warm up graph by calling model once with dummy input
_ = model({
    "input_ids": tf.zeros((1, MAX_LEN), dtype=tf.int32),
    "attention_mask": tf.ones((1, MAX_LEN), dtype=tf.int32),
    "token_type_ids": tf.zeros((1, MAX_LEN), dtype=tf.int32)
})

# load weights if they exist, otherwise train from scratch
weights_path = os.path.join(MODEL_DIR, WEIGHTS_FILE)

if os.path.exists(weights_path):
    print(f"📦 Loading model weights from: {weights_path}")
    model.load_weights(weights_path)
else:
    print("⚠️ Weights not found. Training model from scratch...")

    if 'stance' not in df.columns:
        raise ValueError("❌ 'stance' column is required in the dataset for training.")

    labels = df['stance'].astype(np.float32).values
    label_dataset = tf.data.Dataset.from_tensor_slices(labels).batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((encoded_dataset, label_dataset))

    model.fit(train_dataset, epochs=EPOCHS, verbose=1)

    print(f"💾 Saving trained weights to: {weights_path}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_weights(weights_path)

# generate predictions
print("🔍 Predicting stance scores...")
predictions = model.predict(encoded_dataset, verbose=1).squeeze()

# map regression output to categories
def score_to_category(score: float) -> str:
    if score <= -0.3:
        return "against"
    elif score >= 0.3:
        return "in favor"
    else:
        return "neutral"

results = []
for i, score in enumerate(predictions):
    rounded_score = round(float(score), 1)
    category = score_to_category(rounded_score)

    # convert values to plain Python types for JSON serialization
    title = str(df.iloc[i]["title"])
    abstract = str(df.iloc[i]["abstract"])
    gold = float(df.iloc[i]["stance"]) if "stance" in df.columns else None

    results.append({
        "title": title,
        "abstract": abstract,
        "gold_stance": gold,
        "predicted_stance_score": float(rounded_score),
        "predicted_stance_category": category
    })

# helper for JSON serialization of numpy types
def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

# save results to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=_to_serializable)

print(f"\n✅ Predictions saved to: {OUTPUT_FILE}")
