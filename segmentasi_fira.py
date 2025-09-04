import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segmentasi_wajah_tepat(image_path):
    """
    Segmentasi wajah per zona menggunakan pendekatan anatomi wajah yang tepat
    """
    
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {image_path}")
        return None
    
    # Resize untuk konsistensi
    img = cv2.resize(img, (300, 400))  # Ukuran yang lebih proporsional
    
    # Segmentasi berdasarkan proporsi yang telah disesuaikan
    h, w = img.shape[:2]
    
    # Proporsi yang disesuaikan:
    # - Dahi: 0-25% (dari atas)
    # - Mata: 25-40% (diperkecil area)
    # - Pipi: 40-75% (diperbesar area)
    # - Mulut: 75-100% (dari bawah)
    
    zona = {
        'dahi': img[0:int(h*0.25), :],           # 25% atas
        'mata': img[int(h*0.25):int(h*0.40), :],  # 15% area mata (diperkecil)
        'pipi': img[int(h*0.40):int(h*0.75), :], # 35% area pipi (diperbesar)
        'mulut': img[int(h*0.75):, :]             # 25% bawah
    }
    
    # Konversi ke grayscale untuk analisis
    zona_gray = {k: cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for k, v in zona.items()}
    
    return {
        'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'zona': zona,
        'zona_gray': zona_gray,
        'proporsi': {
            'dahi': '0-30%',
            'mata': '30-60%',
            'pipi': '40-70%',
            'mulut': '70-100%'
        }
    }

def visualisasi_segmentasi_tepat(result, save_path=None):
    """Visualisasi hasil segmentasi yang lebih tepat"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Segmentasi Wajah Per Zona - Pendekatan Anatomi Tepat', fontsize=16)
    
    # Gambar original dengan zona
    ax1 = axes[0, 0]
    img_copy = result['original'].copy()
    h, w = img_copy.shape[:2]
    
    # Gambar garis pembatas zona berdasarkan proporsi anatomi
    y1, y2, y3 = int(h*0.30), int(h*0.60), int(h*0.70)
    
    cv2.line(img_copy, (0, y1), (w, y1), (255, 0, 0), 3)  # Dahi-Mata
    cv2.line(img_copy, (0, y2), (w, y2), (0, 255, 0), 3)  # Mata-Mulut
    cv2.line(img_copy, (0, y3), (w, y3), (0, 0, 255), 3)  # Pipi-Mulut
    
    # Tambah label dengan background
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Background untuk teks
    cv2.rectangle(img_copy, (5, 15), (80, 35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y1+15), (80, y1+35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y2+15), (80, y2+35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y3+15), (80, y3+35), (0, 0, 0), -1)
    
    # Teks label
    cv2.putText(img_copy, 'DAHI', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_copy, 'MATA', (10, y1+30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_copy, 'PIPI', (10, y2+30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img_copy, 'MULUT', (10, y3+30), font, 0.7, (255, 255, 255), 2)
    
    ax1.imshow(img_copy)
    ax1.set_title('Wajah dengan Zona Anatomi Tepat')
    ax1.axis('off')
    
    # Tampilkan setiap zona
    zona_names = ['dahi', 'mata', 'pipi', 'mulut']
    zona_titles = ['Zona Dahi\n(0-30%)', 'Zona Mata\n(30-60%)', 
                   'Zona Pipi\n(40-70%)', 'Zona Mulut\n(70-100%)']
    
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, (nama, title) in enumerate(zip(zona_names, zona_titles)):
        row, col = positions[i]
        ax = axes[row, col]
        ax.imshow(cv2.cvtColor(result['zona'][nama], cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Tampilkan versi grayscale
    for i, (nama, title) in enumerate(zip(zona_names, zona_titles)):
        row, col = positions[i]
        ax = axes[row, col+1] if col < 2 else axes[row+1, 0]
        if row == 1 and col == 1:  # Handle posisi terakhir
            ax = axes[1, 2]
        ax.imshow(result['zona_gray'][nama], cmap='gray')
        ax.set_title(f'{title}\n(Grayscale)', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualisasi disimpan di: {save_path}")
    
    plt.show()

def demo_segmentasi_fira():
    """Demo segmentasi untuk file fira.jpg"""
    
    print("=== DEMO SEGMENTASI WAJAH PER ZONA ===")
    print("Menggunakan pendekatan anatomi wajah yang tepat")
    print("1. Proporsi anatomi manusia")
    print("2. 4 zona utama: Dahi, Mata, Pipi, Mulut")
    print("3. Menggunakan file: fira.jpg")
    
    # Cari file fira.jpg di berbagai lokasi
    possible_paths = [
        'uploads/fira.jpg',
        'dataset/fira.jpg',
        'dataset/high/fira.jpg',
        'dataset/medium/fira.jpg',
        'fira.jpg'
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        print("❌ File fira.jpg tidak ditemukan di lokasi yang diperiksa:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\n⚠️ Menggunakan gambar contoh dari dataset...")
        image_path = 'dataset/high/1.jpg'
    
    print(f"📁 Menggunakan gambar: {image_path}")
    
    # Lakukan segmentasi
    result = segmentasi_wajah_tepat(image_path)
    
    if result:
        # Tampilkan informasi
        print("\n=== INFORMASI SEGMENTASI ===")
        print(f"Gambar: {image_path}")
        print(f"Ukuran: {result['original'].shape}")
        print("\nProporsi zona:")
        for zona, prop in result['proporsi'].items():
            print(f"  {zona}: {prop}")
        
        # Visualisasi
        visualisasi_segmentasi_tepat(result, 'output/segmentation_visualization.png')
        
        # Analisis detail
        print("\n=== ANALISIS ZONA ===")
        for nama, gray_img in result['zona_gray'].items():
            mean = np.mean(gray_img)
            std = np.std(gray_img)
            print(f"{nama.upper()}:")
            print(f"  Mean pixel: {mean:.2f}")
            print(f"  Std dev: {std:.2f}")
            print(f"  Ukuran: {gray_img.shape}")
        
        print("\n✅ Segmentasi selesai!")
        print("📊 Hasil visualisasi: output/segmentation_visualization.png")
    
    return result

if __name__ == "__main__":
    demo_segmentasi_fira()
