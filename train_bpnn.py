import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np

def augment_image(img):
    aug_imgs = [img]
    # Flip horizontal
    aug_imgs.append(cv2.flip(img, 1))
    # Flip vertical
    aug_imgs.append(cv2.flip(img, 0))
    # Rotate 15 degrees
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1)
    aug_imgs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    # Rotate -15 degrees
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -15, 1)
    aug_imgs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    # Brightness up
    aug_imgs.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))
    # Brightness down
    aug_imgs.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-20))
    return aug_imgs

from pso import PSO
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from plot_pso_fitness import plot_fitness_convergence

def fitness_function(params, X_train, y_train, X_val, y_val):
    # params is an array representing hidden layer sizes, e.g. [16, 8]
    # Convert params to tuple of ints for hidden_layer_sizes
    hidden_layers = tuple(int(max(1, round(p))) for p in params)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    # We want to minimize fitness, so return 1 - accuracy
    return 1 - accuracy

def train_model_with_pso():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs("output", exist_ok=True)
    train_data = pd.read_csv("dataset/keriput_train.csv")
    test_data = pd.read_csv("dataset/keriput_test.csv")
    print('Distribusi label di dataset:')
    print(train_data['label'].value_counts())
    fitur_names = ['dahi', 'mata', 'pipi', 'mulut']
    if 'jumlah_kontur' in train_data.columns and 'panjang_total_kontur' in train_data.columns:
        fitur_names += ['jumlah_kontur', 'panjang_total_kontur']
    for fitur in fitur_names:
        plt.figure(figsize=(6,4))
        for label in train_data['label'].unique():
            subset = train_data[train_data['label'] == label]
            plt.hist(subset[fitur], bins=20, alpha=0.5, label=label)
        plt.title(f"Distribusi Fitur: {fitur}")
        plt.xlabel(fitur)
        plt.ylabel("Jumlah")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"output/hist_{fitur}.png")
        plt.close()
        desc = train_data.groupby('label')[fitur].describe()
        desc.to_csv(f"output/stat_{fitur}.csv")

    X_train = []
    y_train = []
    for idx, row in train_data.iterrows():
        fitur = [row[nama] for nama in fitur_names]
        label = row['label']
        for _ in range(6):
            X_train.append(fitur)
            y_train.append(label)

    X_test = []
    y_test = []
    for idx, row in test_data.iterrows():
        fitur = [row[nama] for nama in fitur_names]
        label = row['label']
        X_test.append(fitur)
        y_test.append(label)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split training data into train and validation for PSO fitness evaluation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scaled, y_train_enc, test_size=0.2, random_state=42)

    # PSO parameters
    dim = 2  # number of hidden layers
    bounds = [1, 50]  # min and max neurons per layer
    pso = PSO(lambda params: fitness_function(params, X_train_sub, y_train_sub, X_val, y_val),
              dim=dim, bounds=bounds, num_particles=20, max_iter=30)

    best_position, best_score, fitness_history = pso.optimize()
    best_hidden_layers = tuple(int(max(1, round(p))) for p in best_position)
    print(f"Best hidden layer sizes found by PSO: {best_hidden_layers} with fitness {best_score}")

    # Plot fitness convergence curve
    plot_fitness_convergence(fitness_history)

    # Train final model with best parameters on full training data
    final_mlp = MLPClassifier(hidden_layer_sizes=best_hidden_layers, max_iter=1000, random_state=42)
    final_mlp.fit(X_train_scaled, y_train_enc)
    y_pred = final_mlp.predict(X_test_scaled)

    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Final model accuracy on test set: {acc:.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

    os.makedirs("model", exist_ok=True)
    joblib.dump(final_mlp, 'model/bpnn_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    print("Final model saved to model/")

    cm = confusion_matrix(y_test_enc, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {best_hidden_layers}")
    plt.savefig(f"output/confusion_matrix_{best_hidden_layers}.png")
    plt.close()

def load_model(path):
    return joblib.load(path)

if __name__ == "__main__":
    train_model_with_pso()
