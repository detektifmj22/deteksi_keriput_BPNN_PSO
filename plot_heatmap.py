import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

os.makedirs('output', exist_ok=True)

parser = argparse.ArgumentParser(description='Generate heatmaps for training or testing data.')
parser.add_argument('--data', choices=['train', 'test', 'all'], default='all', help='Pilih dataset: train, test, atau all (default)')
args = parser.parse_args()

if args.data == 'train':
    csv_path = 'dataset/keriput_train.csv'
elif args.data == 'test':
    csv_path = 'dataset/keriput_test.csv'
else:
    csv_path = 'dataset/keriput_dataset.csv'

df = pd.read_csv(csv_path)
fitur = ['dahi', 'mata', 'pipi', 'mulut']
label = 'label'

corr = df[fitur].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title(f'Korelasi Antar Fitur ({args.data})')
plt.tight_layout()
plt.savefig(f'output/heatmap_korelasi_fitur_{args.data}.png')
plt.close()

pivot = df.groupby(label)[fitur].mean()
plt.figure(figsize=(6,3))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title(f'Rata-rata Nilai Fitur per Label ({args.data})')
plt.tight_layout()
plt.savefig(f'output/heatmap_fitur_per_label_{args.data}.png')
plt.close()
