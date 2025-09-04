import os
import shutil
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def save_misclassified_samples(model_path='model/bpnn_model.pkl',
                               scaler_path='model/scaler.pkl',
                               label_encoder_path='model/label_encoder.pkl',
                               test_csv='dataset/keriput_test.csv',
                               dataset_dir='dataset',
                               output_dir='output/visualization_example'):
    os.makedirs(output_dir, exist_ok=True)

    # Load model, scaler, label encoder
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    # Load test data
    test_data = pd.read_csv(test_csv)

    fitur_names = ['dahi', 'mata', 'pipi', 'mulut']
    if 'jumlah_kontur' in test_data.columns and 'panjang_total_kontur' in test_data.columns:
        fitur_names += ['jumlah_kontur', 'panjang_total_kontur']

    X_test = []
    y_test = []
    image_paths = []

    for idx, row in test_data.iterrows():
        fitur = [row[nama] for nama in fitur_names]
        label = row['label']
        X_test.append(fitur)
        y_test.append(label)
        # Assuming image filename is in a column 'image' or construct path from label and index
        # Here we try to find image in dataset/<label> folder with index or filename column
        if 'image' in test_data.columns:
            image_paths.append(os.path.join(dataset_dir, row['image']))
        else:
            # fallback: try to find image by index in label folder
            label_folder = os.path.join(dataset_dir, label)
            # Try to find any jpg file in label folder (simplified)
            files = [f for f in os.listdir(label_folder) if f.lower().endswith('.jpg')]
            if files:
                image_paths.append(os.path.join(label_folder, files[0]))
            else:
                image_paths.append(None)

    X_test = np.array(X_test)
    y_test_enc = label_encoder.transform(y_test)
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    misclassified_indices = [i for i, (yt, yp) in enumerate(zip(y_test_enc, y_pred)) if yt != yp]

    for i in misclassified_indices:
        true_label = label_encoder.inverse_transform([y_test_enc[i]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
        print(f"Misclassified sample index {i}: true label = {true_label}, predicted = {pred_label}")
        img_path = image_paths[i]
        if img_path and os.path.exists(img_path):
            dest_folder = os.path.join(output_dir, f"{true_label}_misclassified_as_{pred_label}")
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)

    print(f"Disimpan contoh sampel yang salah klasifikasi di folder {output_dir}")

if __name__ == "__main__":
    save_misclassified_samples()
