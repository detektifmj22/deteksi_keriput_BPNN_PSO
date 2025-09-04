import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segmentasi_wajah_perbaikan(image_path):
    """
    Segmentasi wajah per zona dengan proporsi yang telah disesuaikan
    - Area mata diperkecil
    - Area pipi diperbesar
    """
    
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {image_path}")
        return None
    
    # Resize untuk konsistensi
    img = cv2.resize(img, (300, 400))
    
    # Proporsi yang telah disesuaikan:
    # - Dahi: 0-25% (dari atas)
    # - Mata: 25-40% (diperkecil dari 30-60% menjadi 15% area)
    # - Pipi: 40-75% (diperbesar dari 40-70% menjadi 35% area)
    # - Mulut: 75-100% (dari bawah)
    
    h, w = img.shape[:2]
    
    zona = {
        'dahi': img[0:int(h*0.35), :],           # 25% atas
        'mata': img[int(h*0.35):int(h*0.50), :],  # 15% area mata (diperkecil)
        'pipi': img[int(h*0.50):int(h*0.75), :], # 35% area pipi (diperbesar)
        'mulut': img[int(h*0.75):, :]             # 25% bawah
    }
    
    # Konversi ke grayscale untuk analisis
    zona_gray = {k: cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for k, v in zona.items()}
    
    return {
        'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'zona': zona,
        'zona_gray': zona_gray,
        'proporsi': {
            'dahi': '0-35%',
            'mata': '35-50%',
            'pipi': '50-75%',
            'mulut': '75-100%'
        }
    }

def visualisasi_segmentasi_perbaikan(result, save_path=None):
    """Visualisasi hasil segmentasi dengan proporsi yang telah disesuaikan"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Segmentasi Wajah Per Zona - Proporsi Disesuaikan', fontsize=16)
    
    # Gambar original dengan zona
    ax1 = axes[0, 0]
    img_copy = result['original'].copy()
    h, w = img_copy.shape[:2]
    
    # Gambar garis pembatas zona yang telah disesuaikan
    y1, y2, y3 = int(h*0.35), int(h*0.50), int(h*0.75)
    
    cv2.line(img_copy, (0, y1), (w, y1), (255, 0, 0), 3)  # Dahi-Mata
    cv2.line(img_copy, (0, y2), (w, y2), (0, 255, 0), 3)  # Mata-Pipi
    cv2.line(img_copy, (0, y3), (w, y3), (0, 0, 255), 3)  # Pipi-Mulut
    
    # Tambah label dengan background
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Background untuk teks
    cv2.rectangle(img_copy, (5, 15), (100, 35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y1+15), (100, y1+35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y2+15), (100, y2+35), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (5, y3+15), (100, y3+35), (0, 0, 0), -1)
    
    # Teks label dengan proporsi baru
    cv2.putText(img_copy, 'DAHI (0-25%)', (10, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img_copy, 'MATA (25-40%)', (10, y1+30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img_copy, 'PIPI (40-75%)', (10, y2+30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img_copy, 'MULUT (75-100%)', (10, y3+30), font, 0.6, (255, 255, 255), 2)
    
    ax1.imshow(img_copy)
    ax1.set_title('Wajah dengan Zona yang Disesuaikan\n(Mata diperkecil, Pipi diperbesar)')
    ax1.axis('off')
    
    # Tampilkan setiap zona
    zona_names = ['dahi', 'mata', 'pipi', 'mulut']
    zona_titles = ['Zona Dahi\n(0-35%)', 'Zona Mata\n(25-40%)', 
                   'Zona Pipi\n(50-75%)', 'Zona Mulut\n(75-100%)']
    
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

def demo_segmentasi_fira_perbaikan():
    """Demo segmentasi untuk file fira.jpg dengan proporsi yang telah disesuaikan"""
    
    print("=== SEGMENTASI WAJAH PER ZONA - PROPORSI DISESUAIKAN ===")
    print("Perubahan proporsi:")
    print("✅ Area mata diperkecil: dari 30% menjadi 15%")
    print("✅ Area pipi diperbesar: dari 30% menjadi 35%")
    print("✅ Menggunakan file: fira.jpg")
    
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
    result = segmentasi_wajah_perbaikan(image_path)
    
    if result:
        # Tampilkan informasi
        print("\n=== INFORMASI SEGMENTASI ===")
        print(f"Gambar: {image_path}")
        print(f"Ukuran: {result['original'].shape}")
        print("\nProporsi zona yang telah disesuaikan:")
        for zona, prop in result['proporsi'].items():
            print(f"  {zona}: {prop}")
        
        # Visualisasi
        # Tampilkan gambar original dan zona grayscale dalam satu baris
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes[0].imshow(result['original'])
        axes[0].set_title('Gambar Original')
        axes[0].axis('off')
        
        zona_names = ['dahi', 'mata', 'pipi', 'mulut']
        for i, zona_name in enumerate(zona_names):
            axes[i+1].imshow(result['zona_gray'][zona_name], cmap='gray')
            axes[i+1].set_title(f'Zona {zona_name.capitalize()} (Grayscale)')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/segmentation_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== ANALISIS ZONA ===")
        for nama, gray_img in result['zona_gray'].items():
            mean = np.mean(gray_img)
            std = np.std(gray_img)
            print(f"{nama.upper()}:")
            print(f"  Mean pixel: {mean:.2f}")
            print(f"  Std dev: {std:.2f}")
            print(f"  Ukuran: {gray_img.shape}")
            print(f"  Area: {len(gray_img) * len(gray_img[0])} pixels")
        
        print("\n✅ Segmentasi selesai dengan proporsi yang telah disesuaikan!")
        print("📊 Hasil visualisasi: output/segmentation_visualization.png")
    
    return result

if __name__ == "__main__":
    demo_segmentasi_fira_perbaikan()
