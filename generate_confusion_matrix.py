import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

def plot_and_save_confusion_matrix(y_true_enc, y_pred_enc, label_encoder, output_path):
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix BPNN-PSO Model')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    test_csv = 'dataset/keriput_test.csv'
    model_path = 'model/bpnn_model.pkl'
    scaler_path = 'model/scaler.pkl'
    label_encoder_path = 'model/label_encoder.pkl'
    output_path = 'output/confusion_matrix_bpnn_pso.png'

    X_scaled, y_true, y_true_enc, model, label_encoder = load_data_and_model(test_csv, model_path, scaler_path, label_encoder_path)
    y_pred = model.predict(X_scaled)

    if label_encoder:
        print("Label encoder classes:", label_encoder.classes_)
        print("Predicted labels (y_pred):", y_pred)
        # y_pred is already numeric, no need to transform
        y_pred_enc = y_pred
    else:
        y_pred_enc = y_pred

    plot_and_save_confusion_matrix(y_true_enc, y_pred_enc, label_encoder, output_path)
