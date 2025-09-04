import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_bar_mean_by_label(csv_file, feature_name, output_dir='output'):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(8, 5))
    # Plot mean values grouped by label
    plt.bar(df['label'], df['mean'], color='skyblue', edgecolor='black')
    plt.title(f'Mean {feature_name} per Label')
    plt.xlabel('Label')
    plt.ylabel(f'Mean {feature_name}')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'mean_{feature_name}_per_label.png')
    plt.savefig(output_path)
    plt.close()
    print(f'Bar chart mean {feature_name} per label disimpan di {output_path}')

if __name__ == "__main__":
    csv_files_features = {
        'output/stat_dahi.csv': 'dahi',
        'output/stat_mata.csv': 'mata',
        'output/stat_pipi.csv': 'pipi',
        'output/stat_mulut.csv': 'mulut',
        'output/stat_jumlah_kontur.csv': 'jumlah_kontur',
        'output/stat_panjang_total_kontur.csv': 'panjang_total_kontur'
    }

    for csv_file, feature in csv_files_features.items():
        plot_bar_mean_by_label(csv_file, feature)
