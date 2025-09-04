import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('dataset/keriput_dataset.csv')

# Pastikan dataset memiliki kolom jumlah_kontur dan panjang_total_kontur
required_features = ['dahi', 'mata', 'pipi', 'mulut', 'jumlah_kontur', 'panjang_total_kontur']
for col in required_features:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset. Pastikan ekstraksi fitur sudah benar.")

X = df[required_features].values
y_true = df['label'].values

try:
    scaler = joblib.load('model/scaler.pkl')
    X_scaled = scaler.transform(X)
except:
    scaler = None
    X_scaled = X

model = joblib.load('model/bpnn_model.pkl')
try:
    label_encoder = joblib.load('model/label_encoder.pkl')
    y_true_enc = label_encoder.transform(y_true)
except:
    label_encoder = None
    y_true_enc = y_true

y_pred = model.predict(X_scaled)


if label_encoder is not None:
    y_pred_label = label_encoder.inverse_transform(y_pred)
    y_true_label = y_true
else:
    y_pred_label = y_pred
    y_true_label = y_true

print('Akurasi:', accuracy_score(y_true_label, y_pred_label))
print('\nConfusion Matrix:')
print(confusion_matrix(y_true_label, y_pred_label))
print('\nClassification Report (precision, recall, f1-score):')
print(classification_report(y_true_label, y_pred_label, digits=4))
