import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Rectangle

def create_wrinkle_histograms(dataset_path='dataset/keriput_dataset.csv', output_dir='output'):
    """
    Create histograms for wrinkle count and wrinkle length distribution post-preprocessing
    
    Parameters:
    -----------
    dataset_path : str
        Path to the preprocessed dataset CSV file
    output_dir : str
        Directory to save the histogram plots
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Histogram of Wrinkle Count and Wrinkle Length Distribution Post-Preprocessing', 
                 fontsize=16, fontweight='bold')
    
    # 1. Wrinkle Count Distribution
    ax1 = axes[0, 0]
    n_bins_count = 30
    
    # Overall distribution
    ax1.hist(df['jumlah_kontur'], bins=n_bins_count, alpha=0.7, color='#3498db', 
             edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_count = df['jumlah_kontur'].mean()
    median_count = df['jumlah_kontur'].median()
    std_count = df['jumlah_kontur'].std()
    
    ax1.axvline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    ax1.axvline(median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
    
    ax1.set_xlabel('Wrinkle Count (Number of Contours)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Wrinkle Count', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Wrinkle Length Distribution
    ax2 = axes[0, 1]
    n_bins_length = 30
    
    # Overall distribution
    ax2.hist(df['panjang_total_kontur'], bins=n_bins_length, alpha=0.7, color='#e74c3c', 
             edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_length = df['panjang_total_kontur'].mean()
    median_length = df['panjang_total_kontur'].median()
    std_length = df['panjang_total_kontur'].std()
    
    ax2.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax2.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
    
    ax2.set_xlabel('Wrinkle Length (Total Contour Length)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Wrinkle Length', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Wrinkle Count by Label
    ax3 = axes[1, 0]
    
    # Create box plot for wrinkle count by label
    box_plot = ax3.boxplot([df[df['label'] == label]['jumlah_kontur'] for label in df['label'].unique()], 
                           labels=df['label'].unique(), patch_artist=True)
    
    # Color the boxes
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Wrinkle Severity Label', fontsize=12)
    ax3.set_ylabel('Wrinkle Count', fontsize=12)
    ax3.set_title('Wrinkle Count Distribution by Severity Label', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Wrinkle Length by Label
    ax4 = axes[1, 1]
    
    # Create box plot for wrinkle length by label
    box_plot2 = ax4.boxplot([df[df['label'] == label]['panjang_total_kontur'] for label in df['label'].unique()], 
                            labels=df['label'].unique(), patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Wrinkle Severity Label', fontsize=12)
    ax4.set_ylabel('Wrinkle Length', fontsize=12)
    ax4.set_title('Wrinkle Length Distribution by Severity Label', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'wrinkle_histograms_post_preprocessing.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual detailed histograms
    create_detailed_histograms(df, output_dir)
    
    print(f"Histograms saved to: {output_path}")
    return df

def create_detailed_histograms(df, output_dir):
    """Create detailed individual histograms with KDE plots"""
    
    # Wrinkle Count with KDE
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['jumlah_kontur'], kde=True, color='#3498db', bins=30)
    plt.axvline(df['jumlah_kontur'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["jumlah_kontur"].mean():.1f}')
    plt.axvline(df['jumlah_kontur'].median(), color='green', linestyle='--', 
                label=f'Median: {df["jumlah_kontur"].median():.1f}')
    plt.title('Wrinkle Count Distribution with KDE')
    plt.xlabel('Wrinkle Count')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['panjang_total_kontur'], kde=True, color='#e74c3c', bins=30)
    plt.axvline(df['panjang_total_kontur'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["panjang_total_kontur"].mean():.1f}')
    plt.axvline(df['panjang_total_kontur'].median(), color='green', linestyle='--', 
                label=f'Median: {df["panjang_total_kontur"].median():.1f}')
    plt.title('Wrinkle Length Distribution with KDE')
    plt.xlabel('Wrinkle Length')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    detailed_path = os.path.join(output_dir, 'wrinkle_histograms_detailed.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create log-scale histograms for better visualization of skewed data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['jumlah_kontur'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    plt.yscale('log')
    plt.title('Wrinkle Count Distribution (Log Scale)')
    plt.xlabel('Wrinkle Count')
    plt.ylabel('Log Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['panjang_total_kontur'], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
    plt.yscale('log')
    plt.title('Wrinkle Length Distribution (Log Scale)')
    plt.xlabel('Wrinkle Length')
    plt.ylabel('Log Frequency')
    
    plt.tight_layout()
    log_scale_path = os.path.join(output_dir, 'wrinkle_histograms_log_scale.png')
    plt.savefig(log_scale_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n=== WRINKLE FEATURE STATISTICS ===")
    print(f"\nWrinkle Count:")
    print(f"  Mean: {df['jumlah_kontur'].mean():.2f}")
    print(f"  Median: {df['jumlah_kontur'].median():.2f}")
    print(f"  Std Dev: {df['jumlah_kontur'].std():.2f}")
    print(f"  Min: {df['jumlah_kontur'].min()}")
    print(f"  Max: {df['jumlah_kontur'].max()}")
    
    print(f"\nWrinkle Length:")
    print(f"  Mean: {df['panjang_total_kontur'].mean():.2f}")
    print(f"  Median: {df['panjang_total_kontur'].median():.2f}")
    print(f"  Std Dev: {df['panjang_total_kontur'].std():.2f}")
    print(f"  Min: {df['panjang_total_kontur'].min()}")
    print(f"  Max: {df['panjang_total_kontur'].max()}")
    
    print(f"\nBy Label:")
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        print(f"\n{label.upper()}:")
        print(f"  Count - Mean: {label_data['jumlah_kontur'].mean():.2f}, Median: {label_data['jumlah_kontur'].median():.2f}")
        print(f"  Length - Mean: {label_data['panjang_total_kontur'].mean():.2f}, Median: {label_data['panjang_total_kontur'].median():.2f}")

if __name__ == "__main__":
    # Generate all histograms
    df = create_wrinkle_histograms()
    print("\n✅ Histogram generation completed successfully!")
    print("📊 Check the 'output' directory for generated plots:")
    print("   - wrinkle_histograms_post_preprocessing.png")
    print("   - wrinkle_histograms_detailed.png")
    print("   - wrinkle_histograms_log_scale.png")
