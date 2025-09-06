"""
This script was generated with the assistance of ChatGPT based on the following
kinds of prompts/instructions:

1. General request / purpose
   - "Fine-tune SciBERT (via HuggingFace Transformers and TensorFlow/Keras) as a regression
      model to predict stance scores from research article abstracts and titles."

2. Data handling
   - "Load input data from a JSON file with 'title', 'abstract', and 'stance' fields.
      Preprocess the text and labels, and split the dataset into training and validation sets."

3. Tokenization
   - "Use the SciBERT tokenizer to convert texts into BERT input IDs, attention masks,
      and token type IDs."

4. Sample weighting
   - "Apply Kernel Density Estimation to compute sample weights that rebalance the
      training set distribution and scale them into a defined range."

5. Model architecture
   - "Wrap the TFBertModel (SciBERT) with a regression head (Dense layer, tanh activation)
      to produce a stance score in [-1.0, 1.0]."

6. Training procedure
   - "Train with MSE loss and Adam optimizer, use EarlyStopping and ModelCheckpoint callbacks,
      and evaluate on the validation set."

7. Saving
   - "Save trained weights and tokenizer into a timestamped output directory."
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from time import strftime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

# configuration
DATA_FILE     = Path("./NLP_short.json")
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO  = 0.06  # defined but not used unless a custom scheduler is added
VAL_SPLIT     = 0.2
MAX_LEN       = 128

# create output directory (timestamped)
OUT_DIR = Path("TrainModel") / strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# check GPU availability
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("✅ GPU detected:", gpus[0].name)
else:
    print("⚠️ No GPU detected — running on CPU")

# load dataset (must contain 'title', 'abstract', 'stance')
if not DATA_FILE.exists():
    raise FileNotFoundError(f"❌ File not found: {DATA_FILE}")

df = pd.read_json(DATA_FILE).dropna(subset=["title", "abstract", "stance"])
texts = df.apply(lambda row: f"{row['title']} [SEP] {row['abstract']}", axis=1).tolist()
labels = df["stance"].values.astype(np.float32)

# tokenize texts with SciBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
encodings = tokenizer(
    texts,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="tf"
)

# compute sample weights using Kernel Density Estimation to handle imbalanced stance distribution
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(labels.reshape(-1, 1))
log_dens = kde.score_samples(labels.reshape(-1, 1))
inv_dens = 1.0 / (np.exp(log_dens) + 1e-6)  # inverse density
weights = MinMaxScaler((1.0, 10.0)).fit_transform(inv_dens.reshape(-1, 1)).flatten()

# split into train/validation
train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=VAL_SPLIT, random_state=42)

train_inputs = {k: tf.gather(v, train_idx) for k, v in encodings.items()}
val_inputs   = {k: tf.gather(v, val_idx)   for k, v in encodings.items()}
train_labels = tf.convert_to_tensor(labels[train_idx], dtype=tf.float32)
val_labels   = tf.convert_to_tensor(labels[val_idx], dtype=tf.float32)
train_weights = tf.convert_to_tensor(weights[train_idx], dtype=tf.float32)
val_weights   = tf.convert_to_tensor(weights[val_idx], dtype=tf.float32)

# build regression model using SciBERT backbone + tanh regression head
bert_model = TFBertModel.from_pretrained("allenai/scibert_scivocab_uncased", from_pt=True)

input_ids      = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
token_type_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="token_type_ids")

bert_outputs = bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids
)

pooled_output = bert_outputs.pooler_output  # CLS embedding

# regression head
output = Dense(1, activation="tanh", name="stance_score")(pooled_output)

model = Model(
    inputs={"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
    outputs=output
)

# compile with Adam optimizer, MSE loss, and MAE metric for monitoring
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

# print model summary with parameter counts
model.summary()

# create TensorFlow datasets with sample weights
train_ds = Dataset.from_tensor_slices((train_inputs, train_labels, train_weights)).shuffle(1000).batch(BATCH_SIZE)
val_ds   = Dataset.from_tensor_slices((val_inputs, val_labels, val_weights)).batch(BATCH_SIZE)

# set up callbacks: early stopping and checkpointing
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(OUT_DIR / "checkpoint.weights.h5"), save_best_only=True)
]

# train the model
history = model.fit(
    train_ds.map(lambda x, y, w: (x, y, w)),   # Keras understands sample weights as third argument
    validation_data=val_ds.map(lambda x, y, w: (x, y, w)),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# evaluate on validation set (ignore weights during evaluation)
loss, mae = model.evaluate(val_ds.map(lambda x, y, w: (x, y)), verbose=1)
print(f"\n🎯 Final MAE: {mae:.4f}")

# save trained weights and tokenizer
model.save_weights(str(OUT_DIR / "scibert_regression.weights.h5"))
tokenizer.save_pretrained(OUT_DIR)

print(f"✅ Model and tokenizer saved to: {OUT_DIR}")
