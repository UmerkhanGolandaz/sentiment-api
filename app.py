"""
Flask API for Sentiment Analysis Model
Serves predictions and model stats to React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
models_data = {}
tfidf = None
model_stats = {}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_models():
    global models_data, tfidf, model_stats

    print("Loading dataset...")
    df = pd.read_csv('movie_reviews.csv')

    # Already filtered to ratings 1-4 and 7-10
    df['sentiment'] = (df['Ratings'] >= 7).astype(int)
    df_model = df.copy()

    # Clean text
    df_model['clean_reviews'] = df_model['Reviews'].apply(clean_text)
    df_model = df_model[df_model['clean_reviews'].str.len() >= 10].reset_index(drop=True)

    sentiment_counts = df_model['sentiment'].value_counts()
    sentiment_pct = df_model['sentiment'].value_counts(normalize=True) * 100

    X = df_model['clean_reviews']
    y = df_model['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=10000, stop_words='english',
        ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.95
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train models
    models = {
        'no_reg': LogisticRegression(C=1e9, max_iter=1000, random_state=42, class_weight='balanced'),
        'l2': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42, class_weight='balanced'),
        'l1': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000, random_state=42, class_weight='balanced'),
    }

    model_names_map = {
        'no_reg': 'No Regularization (C=1e9)',
        'l2': 'L2 Regularization (C=1.0)',
        'l1': 'L1 Regularization (C=1.0)'
    }

    results = {}
    for key, model in models.items():
        model.fit(X_train_tfidf, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))
        test_acc = accuracy_score(y_test, model.predict(X_test_tfidf))
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        report = classification_report(y_test, y_pred,
                                       target_names=['Negative', 'Positive'],
                                       output_dict=True)

        results[key] = {
            'name': model_names_map[key],
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'overfit_gap': round(train_acc - test_acc, 4),
            'confusion_matrix': cm,
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(roc_auc, 4)},
            'report': report
        }
        print(f"  {model_names_map[key]}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    models_data = {'l2': models['l2']}

    # Sparsity analysis
    l1_coefs = models['l1'].coef_[0]
    l2_coefs = models['l2'].coef_[0]
    no_reg_coefs = models['no_reg'].coef_[0]

    l1_zeros = int(np.sum(np.abs(l1_coefs) < 1e-6))
    l2_zeros = int(np.sum(np.abs(l2_coefs) < 1e-6))
    no_reg_zeros = int(np.sum(np.abs(no_reg_coefs) < 1e-6))
    total_features = len(l1_coefs)

    # Top features
    feature_names = tfidf.get_feature_names_out()
    l2_model = models['l2']
    top_pos_idx = np.argsort(l2_model.coef_[0])[-15:][::-1]
    top_neg_idx = np.argsort(l2_model.coef_[0])[:15]

    top_positive = [{'word': feature_names[i], 'coef': round(float(l2_model.coef_[0][i]), 4)} for i in top_pos_idx]
    top_negative = [{'word': feature_names[i], 'coef': round(float(l2_model.coef_[0][i]), 4)} for i in top_neg_idx]

    # Hyperparameter tuning
    C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    tuning = []
    best_acc = 0
    best_C = 1.0
    for C in C_values:
        lr = LogisticRegression(penalty='l2', C=C, max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train_tfidf, y_train)
        tr = accuracy_score(y_train, lr.predict(X_train_tfidf))
        te = accuracy_score(y_test, lr.predict(X_test_tfidf))
        tuning.append({'C': C, 'train': round(tr, 4), 'test': round(te, 4)})
        if te > best_acc:
            best_acc = te
            best_C = C

    # Use best model for predictions
    best_model = LogisticRegression(penalty='l2', C=best_C, max_iter=1000, random_state=42, class_weight='balanced')
    best_model.fit(X_train_tfidf, y_train)
    models_data['best'] = best_model

    model_stats = {
        'dataset': {
            'total_reviews': len(df),
            'clean_reviews': len(df_model),
            'removed_neutral': removed,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': int(X_train_tfidf.shape[1]),
            'columns': df.columns.tolist(),
            'num_columns': len(df.columns)
        },
        'bias': {
            'emotions': {k: int(v) for k, v in emotion_counts.items()},
            'emotion_pct': {k: round(v, 1) for k, v in emotion_pct.items()},
            'imbalance_ratio': round(imbalance_ratio, 1),
            'sentiment_counts': {'negative': int(sentiment_counts[0]), 'positive': int(sentiment_counts[1])},
            'sentiment_pct': {'negative': round(sentiment_pct[0], 1), 'positive': round(sentiment_pct[1], 1)},
            'ratings': {str(int(k)): int(v) for k, v in df['Ratings'].value_counts().sort_index().items()}
        },
        'models': results,
        'sparsity': {
            'total_features': total_features,
            'no_reg_zeros': no_reg_zeros,
            'l2_zeros': l2_zeros,
            'l1_zeros': l1_zeros
        },
        'top_features': {
            'positive': top_positive,
            'negative': top_negative
        },
        'tuning': {
            'results': tuning,
            'best_C': best_C,
            'best_acc': round(best_acc, 4)
        }
    }

    print(f"\nModels trained! Best: L2 C={best_C}, Accuracy={best_acc:.4f}")


@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({'error': 'Model is still loading, please wait 30 seconds...'}), 503

    data = request.json
    review = data.get('review', '')
    if not review.strip():
        return jsonify({'error': 'Please enter a review'}), 400

    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])

    model = models_data['best']
    prediction = int(model.predict(vectorized)[0])
    proba = model.predict_proba(vectorized)[0]

    return jsonify({
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': round(float(max(proba)) * 100, 1),
        'positive_prob': round(float(proba[1]) * 100, 1),
        'negative_prob': round(float(proba[0]) * 100, 1),
        'cleaned_text': cleaned,
        'word_count': len(cleaned.split())
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify(model_stats)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': len(models_data) > 0})


import threading

model_ready = False

def train_in_background():
    global model_ready
    train_models()
    model_ready = True

# Start training in background so server binds port immediately
threading.Thread(target=train_in_background, daemon=True).start()


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'ready': model_ready})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nAPI running at http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
