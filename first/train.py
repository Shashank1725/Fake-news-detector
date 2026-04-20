import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
import lime
import lime.lime_text

#1. Load data
print("Loading data...")
bf_fake = pd.read_csv(r"C:\MLProject\BuzzFeed_fake_news_content.csv")
bf_real = pd.read_csv(r"C:\MLProject\BuzzFeed_real_news_content.csv")
pf_fake = pd.read_csv(r"C:\MLProject\PolitiFact_fake_news_content.csv")
pf_real = pd.read_csv(r"C:\MLProject\PolitiFact_real_news_content.csv")
t_true1=pd.read_csv(r"C:\MLProject\True1.csv")
t_true=pd.read_csv(r"C:\MLProject\True.csv")
f_fake=pd.read_csv(r"C:\MLProject\Fake.csv")
f_fake1=pd.read_csv(r"C:\MLProject\Fake1.csv")

bf_fake['label'] = 1
pf_fake['label'] = 1
f_fake['label'] = 1
f_fake1['label'] = 1
t_true['label'] = 0
t_true1['label'] = 0
bf_real['label'] = 0
pf_real['label'] = 0

data = pd.concat([bf_fake, bf_real, pf_fake, pf_real,t_true,t_true1,f_fake,f_fake1], ignore_index=True)
print(f"Dataset shape: {data.shape}")
print(data['label'].value_counts())

# 2. Clean data
data.dropna(subset=['text'], inplace=True)
data['text'] = data['text'].fillna('')
data['title'] = data['title'].fillna('')

# 3. Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # remove punctuation/numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w)>2]
    return ' '.join(tokens)

print("Cleaning text (this may take a moment)...")
data['clean_text'] = data['text'].apply(clean_text)

#  4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label'], test_size=0.2, random_state=42
)
print(f"Training samples:{len(X_train)}")
print(f"Testing samples: {len(X_test)}")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#  5. Tokenize & pad
MAX_WORDS = 50000
MAX_LEN   = 500

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

# 6. Build model
print("Building model...")
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#  7. Training
print("Training...")
history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_test_seq, y_test),
    epochs=5,
    batch_size=32
)

#8. Evaluate
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

#  9. Save model & tokenizer
save_dir = os.path.dirname(__file__)
model.save(os.path.join(save_dir, 'model.keras'))
with open(os.path.join(save_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

print("\nSaved model.keras & tokenizer.pkl")
print("You can now run:  streamlit run first/app.py")

# EDA plots
sns.countplot(x='label', data=data)
plt.xticks([0,1], ['Real','Fake'])
plt.title('Real vs Fake article count')
plt.show()

# check average word count
data['word_count'] = data['clean_text'].apply(lambda x: len(x.split()))
print(data.groupby('label')['word_count'].mean())

#10. PassiveAggressive Classifier (baseline)
from sklearn.calibration import CalibratedClassifierCV

# Combine title + text for richer features
data['combined'] = data['title'] + ' ' + data['text']
data['clean_text'] = data['combined'].apply(clean_text)

# Fix class imbalance + convergence
pa_raw = PassiveAggressiveClassifier(max_iter=1000, random_state=42, class_weight='balanced')
pa_model = CalibratedClassifierCV(pa_raw, cv=5)   # wraps it to give real probabilities
pa_model.fit(X_train_tfidf, y_train)
y_pred = pa_model.predict(X_test_tfidf)

#11. Evaluate baseline
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Real','Fake']))
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
    xticklabels=['Real','Fake'],
    yticklabels=['Real','Fake']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))
plt.show()

# Save baseline model & vectorizer
with open(os.path.join(save_dir, 'baseline_model.pkl'), 'wb') as f:
    pickle.dump(pa_model, f)
with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
print("Saved baseline_model.pkl & tfidf_vectorizer.pkl")

# Accuracy/Loss plot for LSTM
fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label="Val Accuracy")
axes[0].set_title('Model accuracy over epochs')
axes[0].set_xlabel('Epochs')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label="Val Loss")
axes[1].set_title('Model loss over epochs')
axes[1].set_xlabel('Epochs')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'accuracy_loss.png'))
plt.show()

# LSTM evaluation
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
y_pred_prob = model.predict(X_test_seq)
y_pred_lstm = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred_lstm, target_names=['Real','Fake']))

# ── LIME Explainability (using TF-IDF + PA baseline — works much better on small datasets) ──
def predict_probab_baseline(texts):
    cleaned = [clean_text(t) for t in texts]
    vectors = tfidf.transform(cleaned)
    return pa_model.predict_proba(vectors)   

explainer = lime.lime_text.LimeTextExplainer(class_names=['Real', 'Fake'])
sample_idx = 5
sample_text = X_test.iloc[sample_idx]
exp = explainer.explain_instance(sample_text, predict_probab_baseline, num_features=10, num_samples=1000)

pa_pred = pa_model.predict(tfidf.transform([clean_text(sample_text)]))[0]
print(f"\nSample text (first 200 chars): {sample_text[:200]}...")
print(f"Prediction: {'FAKE' if pa_pred == 1 else 'REAL'}")
print("\nTop words driving prediction:")
for word, weight in exp.as_list():
    direction = " -> FAKE" if weight > 0 else " -> REAL"
    print(f"  {word}: {weight:+.4f} {direction}")

# Save LIME explanation as HTML
exp.save_to_file(os.path.join(os.path.dirname(__file__), 'lime_explanation.html'))
print("\nLIME explanation saved to first/lime_explanation.html")