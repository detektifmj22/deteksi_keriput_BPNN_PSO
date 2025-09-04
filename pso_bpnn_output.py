import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns
from pso import PSO
from plot_pso_fitness import plot_fitness_convergence

def fitness_function(params, X_train, y_train, X_val, y_val):
    hidden_layers = tuple(int(max(1, round(p))) for p in params)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return 1 - accuracy

def main():
    os.makedirs("output", exist_ok=True)
    train_data = pd.read_csv("dataset/keriput_train.csv")
    test_data = pd.read_csv("dataset/keriput_test.csv")

    fitur_names = ['dahi', 'mata', 'pipi', 'mulut']
    if 'jumlah_kontur' in train_data.columns and 'panjang_total_kontur' in train_data.columns:
        fitur_names += ['jumlah_kontur', 'panjang_total_kontur']

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

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scaled, y_train_enc, test_size=0.2, random_state=42)

    dim = 2
    bounds = [1, 50]
    pso = PSO(lambda params: fitness_function(params, X_train_sub, y_train_sub, X_val, y_val),
              dim=dim, bounds=bounds, num_particles=20, max_iter=30)

    best_position, best_score, fitness_history = pso.optimize()
    best_hidden_layers = tuple(int(max(1, round(p))) for p in best_position)

    # Save and print optimized architecture
    print(f"Optimized BPNN Architecture Resulting from PSO: {best_hidden_layers} with fitness {best_score}")
    with open("output/optimized_bpnn_architecture.txt", "w") as f:
        f.write(f"Optimized BPNN Architecture: {best_hidden_layers}\nFitness: {best_score}\n")

    # Plot and save fitness convergence curve
    plot_fitness_convergence(fitness_history, output_path="output/pso_fitness_convergence.png")

    # Train final model with optimized architecture
    final_mlp = MLPClassifier(hidden_layer_sizes=best_hidden_layers, max_iter=1000, random_state=42)
    final_mlp.fit(X_train_scaled, y_train_enc)
    y_pred = final_mlp.predict(X_test_scaled)

    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Final model accuracy on test set: {acc:.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

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

    # Generate confusion matrix for training data
    y_train_pred = final_mlp.predict(X_train_scaled)
    cm_train = confusion_matrix(y_train_enc, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix Training Data {best_hidden_layers}")
    plt.savefig(f"output/confusion_matrix_training_{best_hidden_layers}.png")
    plt.close()

if __name__ == "__main__":
    main()
