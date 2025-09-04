import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(csv_path, train_ratio=0.8):
    # Load dataset
    df = pd.read_csv(csv_path)
    # Split dataset
    train_df, test_df = train_test_split(df, train_size=train_ratio, stratify=df['label'], random_state=42)
    # Save split datasets
    base_dir = os.path.dirname(csv_path)
    train_path = os.path.join(base_dir, 'keriput_train.csv')
    test_path = os.path.join(base_dir, 'keriput_test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Dataset dibagi menjadi {train_ratio*100}% data pelatihan dan {(1-train_ratio)*100}% data pengujian.")
    print(f"Data pelatihan disimpan di: {train_path}")
    print(f"Data pengujian disimpan di: {test_path}")

if __name__ == "__main__":
    csv_path = "dataset/keriput_dataset.csv"
    split_dataset(csv_path)
