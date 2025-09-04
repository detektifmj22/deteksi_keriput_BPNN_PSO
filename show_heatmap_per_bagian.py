import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def show_heatmap_per_bagian(csv_path):
    """
    Membaca data fitur dari file CSV dan menampilkan heatmap korelasi fitur per bagian wajah.
    """
    df = pd.read_csv(csv_path)
    
    # Pilih kolom fitur yang relevan untuk tiap bagian wajah
    fitur_dahi = ['dahi']
    fitur_mata = ['mata']
    fitur_pipi = ['pipi']
    fitur_mulut = ['mulut']
    
    # Gabungkan semua fitur u/ korelasi
    fitur_all = fitur_dahi + fitur_mata + fitur_pipi + fitur_mulut
    
    # Korelasi antar fitur
    corr = df[fitur_all].corr()
    
    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap Korelasi Fitur Per Bagian Wajah')
    plt.show()

if __name__ == "__main__":
    csv_path = 'dataset/keriput_dataset.csv'
    show_heatmap_per_bagian(csv_path)
