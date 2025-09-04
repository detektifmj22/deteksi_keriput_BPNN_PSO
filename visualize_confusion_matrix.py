import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def load_data_and_model(test_csv_path, model_path, scaler_path=None, label_encoder_path=None):
    df = pd.read_csv(test_csv_path)
    required_features = ['dahi', 'mata', 'pipi', 'mulut', 'jumlah_kontur', 'panjang_total_kontur']
    X = df[required_features].values
    y_true = df['label'].values

    if scaler_path:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    model = joblib.load(model_path)

    if label_encoder_path:
        label_encoder = joblib.load(label_encoder_path)
        y_true_enc = label_encoder.transform(y_true)
    else:
        label_encoder = None
        y_true_enc = y_true

    return X_scaled, y_true, y_true_enc, model, label_encoder

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_csv = 'dataset/keriput_test.csv'
    model_path = 'model/bpnn_model.pkl'
    scaler_path = 'model/scaler.pkl'
    label_encoder_path = 'model/label_encoder.pkl'

    X_scaled, y_true, y_true_enc, model, label_encoder = load_data_and_model(test_csv, model_path, scaler_path, label_encoder_path)

    y_pred = model.predict(X_scaled)

    if label_encoder:
        # Pastikan y_pred bertipe numerik sebelum inverse_transform
        import numpy as np
        if y_pred.dtype.type is np.str_ or y_pred.dtype.type is np.object_:
            y_pred_enc = label_encoder.transform(y_pred)
        else:
            y_pred_enc = y_pred
        y_pred_label = label_encoder.inverse_transform(y_pred_enc)
        y_true_label = y_true
        labels = label_encoder.classes_
    else:
        y_pred_label = y_pred
        y_true_label = y_true
        labels = sorted(set(y_true))

    print('Accuracy:', accuracy_score(y_true_label, y_pred_label))
    print('\nClassification Report:')
    print(classification_report(y_true_label, y_pred_label, digits=4))

    plot_confusion_matrix(y_true_label, y_pred_label, labels, title='Confusion Matrix on Test Data')
