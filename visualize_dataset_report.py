#!/usr/bin/env python3
"""
Script untuk membuat visualisasi lengkap untuk laporan dataset dan preprocessing
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os
from preprocessing import load_and_preprocess_images
from ekstraksi_fitur import ekstrak_fitur_keriput

# Setup direktori output
os.makedirs('output', exist_ok=True)

def create_sample_segmentation_visualization():
    """Membuat visualisasi segmentasi wajah per zona"""
    # Load sample images
    high_images, _ = load_and_preprocess_images('dataset/high')
    medium_images, _ = load_and_preprocess_images('dataset/medium')
    
    # Pilih sample images
    sample_high = high_images[0] if high_images else None
    sample_medium = medium_images[0] if medium_images else None
    
    if sample_high is None or sample_medium is None:
        print("Warning: Sample images not found")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Segmentasi Wajah: Perbandingan High vs Medium Severity', fontsize=16)
    
    # Define zones
    zones = {
        'Original': None,
        'Dahi\n[20:60, 60:140]': (20, 60, 60, 140),
        'Mata\n[70:100, 60:140]': (70, 100, 60, 140),
        'Pipi\n[110:140, 50:150]': (110, 140, 50, 150),
        'Mulut\n[150:180, 70:130]': (150, 180, 70, 130)
    }
    
    zone_names = list(zones.keys())
    
    for row_idx, (sample, label) in enumerate([(sample_high, 'High'), (sample_medium, 'Medium')]):
        for col_idx, zone_name in enumerate(zone_names):
            ax = axes[row_idx, col_idx]
            
            if zone_name == 'Original':
                img_display = sample
                title = f'{label} Severity\n(Original)'
            else:
                y1, y2, x1, x2 = zones[zone_name]
                img_display = sample[y1:y2, x1:x2]
                title = zone_name
            
            ax.imshow(img_display, cmap='gray')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
            # Add rectangle for original image
            if zone_name != 'Original' and col_idx == 0:
                y1, y2, x1, x2 = zones[zone_name]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('output/segmentation_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Visualisasi segmentasi tersimpan: output/segmentation_visualization.png")

def create_class_distribution_plot():
    """Membuat visualisasi distribusi kelas"""
    # Load dataset
    df = pd.read_csv('dataset/keriput_dataset.csv')
    
    # Count classes
    class_counts = df['label'].value_counts()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    colors = ['#ff9999', '#66b3ff']
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors)
    ax1.set_title('Distribusi Kelas dalam Dataset', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Kelas', fontsize=12)
    ax1.set_ylabel('Jumlah Citra', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    # Pie chart
    ax2.pie(class_counts.values, labels=class_counts.index, colors=colors, 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Proporsi Kelas', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Visualisasi distribusi kelas tersimpan: output/class_distribution.png")

def create_augmentation_summary():
    """Membuat ringkasan augmentasi data"""
    # Load dataset
    df = pd.read_csv('dataset/keriput_dataset.csv')
    
    # Original counts
    original_counts = df['label'].value_counts()
    
    # After augmentation (1:6 ratio)
    augmented_counts = original_counts * 6
    
    # Create summary table
    summary_data = {
        'Kelas': original_counts.index,
        'Citra Asli': original_counts.values,
        'Augmentasi per Citra': [6, 6],
        'Total Setelah Augmentasi': augmented_counts.values
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Ringkasan Augmentasi Data', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('output/augmentation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Ringkasan augmentasi tersimpan: output/augmentation_summary.png")

def create_feature_correlation_heatmap():
    """Membuat heatmap korelasi fitur"""
    # Load dataset
    df = pd.read_csv('dataset/keriput_dataset.csv')
    
    # Select numeric features
    numeric_cols = ['dahi', 'mata', 'pipi', 'mulut', 'jumlah_kontur', 'panjang_total_kontur']
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title('Heatmap Korelasi Antar Fitur', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/heatmap_korelasi_fitur_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Heatmap korelasi tersimpan: output/heatmap_korelasi_fitur_all.png")

def create_feature_distribution_plots():
    """Membuat distribusi fitur untuk kedua kelas"""
    # Load dataset
    df = pd.read_csv('dataset/keriput_dataset.csv')
    
    # Features to plot
    features = ['jumlah_kontur', 'panjang_total_kontur']
    
    for feature in features:
        plt.figure(figsize=(12, 5))
        
        # Create subplots
        plt.subplot(1, 2, 1)
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.hist(subset[feature], bins=30, alpha=0.7, label=label, density=True)
        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Density')
        plt.title(f'Distribusi {feature.replace("_", " ").title()}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        df.boxplot(column=feature, by='label', ax=plt.gca())
        plt.title(f'{feature.replace("_", " ").title()} per Kelas')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(f'output/distribusi_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Distribusi {feature} tersimpan: output/distribusi_{feature}.png")

def create_normalization_demo():
    """Membuat demonstrasi efek normalisasi"""
    # Load dataset
    df = pd.read_csv('dataset/keriput_dataset.csv')
    
    # Select a feature
    feature = 'panjang_total_kontur'
    original_data = df[feature]
    
    # Simulate normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original distribution
    ax1.hist(original_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(original_data.mean(), color='red', linestyle='--', label=f'Mean: {original_data.mean():.2f}')
    ax1.axvline(original_data.mean() + original_data.std(), color='orange', linestyle='--', label=f'+1 Std: {original_data.mean() + original_data.std():.2f}')
    ax1.axvline(original_data.mean() - original_data.std(), color='orange', linestyle='--', label=f'-1 Std: {original_data.mean() - original_data.std():.2f}')
    ax1.set_title('Distribusi Fitur Sebelum Normalisasi')
    ax1.set_xlabel('Nilai Fitur')
    ax1.set_ylabel('Frekuensi')
    ax1.legend()
    
    # Normalized distribution
    ax2.hist(normalized_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', label='Mean: 0')
    ax2.axvline(1, color='orange', linestyle='--', label='+1 Std: 1')
    ax2.axvline(-1, color='orange', linestyle='--', label='-1 Std: -1')
    ax2.set_title('Distribusi Fitur Setelah Normalisasi (StandardScaler)')
    ax2.set_xlabel('Nilai Fitur (Z-score)')
    ax2.set_ylabel('Frekuensi')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('output/normalization_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Demo normalisasi tersimpan: output/normalization_demo.png")

def create_label_encoding_demo():
    """Membuat demonstrasi label encoding"""
    # Create encoding table
    encoding_data = {
        'Label Asli': ['High', 'Medium'],
        'Label Setelah Encoding': [1, 0],
        'Deskripsi': ['Keriput Parah/Severe', 'Keriput Sedang/Moderate']
    }
    
    encoding_df = pd.DataFrame(encoding_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=encoding_df.values,
                     colLabels=encoding_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(encoding_df.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Tabel Label Encoding', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('output/label_encoding_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Demo label encoding tersimpan: output/label_encoding_demo.png")

def main():
    """Fungsi utama untuk generate semua visualisasi"""
    print("🚀 Membuat visualisasi lengkap untuk laporan dataset...")
    
    # Create all visualizations
    create_sample_segmentation_visualization()
    create_class_distribution_plot()
    create_augmentation_summary()
    create_feature_correlation_heatmap()
    create_feature_distribution_plots()
    create_normalization_demo()
    create_label_encoding_demo()
    
    print("\n✅ Semua visualisasi telah berhasil dibuat!")
    print("📁 Semua file tersimpan di folder 'output/'")
    print("\nFile yang dihasilkan:")
    print("- output/segmentation_visualization.png")
    print("- output/class_distribution.png")
    print("- output/augmentation_summary.png")
    print("- output/heatmap_korelasi_fitur_all.png")
    print("- output/distribusi_jumlah_kontur.png")
    print("- output/distribusi_panjang_total_kontur.png")
    print("- output/normalization_demo.png")
    print("- output/label_encoding_demo.png")

if __name__ == "__main__":
    main()
