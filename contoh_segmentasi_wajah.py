import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segmentasi_wajah_per_zona(image_path):
    """
    Contoh proses segmentasi wajah per zona menggunakan foto dari dataset
    
    Args:
        image_path: Path ke gambar wajah
    
    Returns:
        dict: Dictionary berisi gambar untuk setiap zona
    """
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak bisa load gambar dari {image_path}")
        return None
    
    # Convert ke RGB untuk visualisasi
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Definisi zona wajah berdasarkan koordinat (disesuaikan dengan ukuran 200x200)
    height, width = img.shape[:2]
    
    # Koordinat zona wajah (disesuaikan dengan ukuran gambar)
    zona = {
        'dahi': img[20:60, 60:140],      # Bagian atas wajah
        'mata': img[70:100, 60:140],     # Area mata
        'pipi': img[110:140, 50:150],    # Area pipi
        'mulut': img[150:180, 70:130]   # Area mulut
    }
    
    # Konversi ke grayscale untuk analisis tekstur
    zona_gray = {
        'dahi': cv2.cvtColor(zona['dahi'], cv2.COLOR_BGR2GRAY) if len(zona['dahi'].shape) == 3 else zona['dahi'],
        'mata': cv2.cvtColor(zona['mata'], cv2.COLOR_BGR2GRAY) if len(zona['mata'].shape) == 3 else zona['mata'],
        'pipi': cv2.cvtColor(zona['pipi'], cv2.COLOR_BGR2GRAY) if len(zona['pipi'].shape) == 3 else zona['pipi'],
        'mulut': cv2.cvtColor(zona['mulut'], cv2.COLOR_BGR2GRAY) if len(zona['mulut'].shape) == 3 else zona['mulut']
    }
    
    return {
        'original': img_rgb,
        'zona': zona,
        'zona_gray': zona_gray,
        'koordinat': {
            'dahi': (20, 60, 60, 140),
            'mata': (70, 100, 60, 140),
            'pipi': (110, 140, 50, 150),
            'mulut': (150, 180, 70, 130)
        }
    }

def visualisasi_segmentasi(result, save_path=None):
    """
    Visualisasi hasil segmentasi wajah per zona
    
    Args:
        result: Hasil dari fungsi segmentasi_wajah_per_zona
        save_path: Path untuk menyimpan visualisasi (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Contoh Segmentasi Wajah Per Zona', fontsize=16)
    
    # Gambar original dengan bounding box zona
    img_copy = result['original'].copy()
    h, w = img_copy.shape[:2]
    
    # Gambar bounding box untuk setiap zona
    for zona_name, (y1, y2, x1, x2) in result['koordinat'].items():
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_copy, zona_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    axes[0, 0].imshow(img_copy)
    axes[0, 0].set_title('Gambar Original dengan Zona')
    axes[0, 0].axis('off')
    
    # Tampilkan setiap zona
    zona_names = ['dahi', 'mata', 'pipi', 'mulut']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, (zona_name, pos) in enumerate(zip(zona_names, positions[:4])):
        if zona_name in result['zona']:
            axes[pos[0], pos[1]].imshow(cv2.cvtColor(result['zona'][zona_name], cv2.COLOR_BGR2RGB) 
                                         if len(result['zona'][zona_name].shape) == 3 
                                         else result['zona_gray'][zona_name], cmap='gray')
            axes[pos[0], pos[1]].set_title(f'Zona {zona_name.capitalize()}')
            axes[pos[1], pos[1]].axis('off')
    
    # Hapus subplot kosong
    if len(zona_names) < 5:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualisasi disimpan di: {save_path}")
    
    plt.show()

def proses_dataset_contoh():
    """
    Contoh proses segmentasi untuk beberapa gambar dari dataset
    """
    # Path ke dataset
    dataset_paths = [
        'dataset/high/1.jpg',
        'dataset/high/2.jpg',
        'dataset/medium/1.jpg',
        'dataset/medium/2.jpg'
    ]
    
    # Cek file yang tersedia
    available_files = []
    for path in dataset_paths:
        if os.path.exists(path):
            available_files.append(path)
    
    if not available_files:
        print("Menggunakan gambar dummy...")
        # Jika file tidak tersedia, gunakan gambar dari preprocessing
        images, filenames = load_and_preprocess_images('dataset/high/')
        if images:
            # Simpan gambar pertama sebagai contoh
            sample_path = 'output/sample_image.jpg'
            cv2.imwrite(sample_path, cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR))
            available_files = [sample_path]
    
    # Proses setiap gambar
    for i, image_path in enumerate(available_files[:2]):  # Proses 2 gambar saja
        print(f"\nProcessing: {image_path}")
        
        # Lakukan segmentasi
        result = segmentasi_wajah_per_zona(image_path)
        
        if result:
            # Visualisasi hasil
            save_path = f'output/segmentasi_contoh_{i+1}.png'
            visualisasi_segmentasi(result, save_path)
            
            # Print informasi zona
            print("Informasi Zona:")
            for zona_name, zona_img in result['zona'].items():
                if zona_img is not None:
                    print(f"  {zona_name}: {zona_img.shape}")
            
            # Ekstrak fitur dari setiap zona (contoh)
            print("\nEkstraksi Fitur dari Zona:")
            for zona_name, zona_gray in result['zona_gray'].items():
                if zona_gray is not None:
                    # Contoh ekstraksi fitur GLCM
                    from skimage.feature import graycomatrix, graycoprops
                    glcm = graycomatrix(zona_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0,0]
                    print(f"  {zona_name} - Contrast: {contrast:.2f}")

if __name__ == "__main__":
    print("=== Contoh Proses Segmentasi Wajah Per Zona ===")
    proses_dataset_contoh()
    print("\nProses selesai! Cek folder 'output/' untuk hasil visualisasi.")
