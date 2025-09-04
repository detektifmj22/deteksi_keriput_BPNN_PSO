import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def plot_classification_report(y_true, y_pred, labels, title='Classification Report'):
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    classes = list(report_dict.keys())[:-3]  # exclude accuracy, macro avg, weighted avg
    precision = [report_dict[cls]['precision'] for cls in classes]
    recall = [report_dict[cls]['recall'] for cls in classes]
    f1_score = [report_dict[cls]['f1-score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1])
    ax.legend()

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

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

if __name__ == "__main__":
    test_csv = 'dataset/keriput_test.csv'
    model_path = 'model/bpnn_model.pkl'
    scaler_path = 'model/scaler.pkl'
    label_encoder_path = 'model/label_encoder.pkl'

    X_scaled, y_true, y_true_enc, model, label_encoder = load_data_and_model(test_csv, model_path, scaler_path, label_encoder_path)

    y_pred = model.predict(X_scaled)

    if label_encoder:
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

    plot_classification_report(y_true_label, y_pred_label, labels, title='Classification Report for BPNN Model')
