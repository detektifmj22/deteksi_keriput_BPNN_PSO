import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segmentasi_wajah(image_path):
    """Segmentasi wajah menjadi 4 zona utama"""
    
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print("Gambar tidak ditemukan")
        return None
    
    # Resize ke ukuran standar
    img = cv2.resize(img, (200, 200))
    
    # Segmentasi zona
    zona = {
        'dahi': img[20:60, 60:140],
        'mata': img[70:100, 60:140],
        'pipi': img[110:140, 50:150],
        'mulut': img[150:180, 70:130]
    }
    
    return img, zona

def tampilkan_hasil(image_path):
    """Tampilkan hasil segmentasi"""
    
    img, zona = segmentasi_wajah(image_path)
    if img is None:
        return
    
    # Buat figure
    plt.figure(figsize=(15, 10))
    
    # Gambar original dengan kotak zona
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Gambar kotak untuk setiap zona
    cv2.rectangle(img_rgb, (60, 20), (140, 60), (255, 0, 0), 2)
    cv2.putText(img_rgb, 'Dahi', (65, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.rectangle(img_rgb, (60, 70), (140, 100), (0, 255, 0), 2)
    cv2.putText(img_rgb, 'Mata', (65, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.rectangle(img_rgb, (50, 110), (150, 140), (0, 0, 255), 2)
    cv2.putText(img_rgb, 'Pipi', (55, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.rectangle(img_rgb, (70, 150), (130, 180), (255, 255, 0), 2)
    cv2.putText(img_rgb, 'Mulut', (75, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Tampilkan gambar original
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Gambar Original dengan Zona')
    plt.axis('off')
    
    # Tampilkan setiap zona
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(zona['dahi'], cv2.COLOR_BGR2RGB))
    plt.title('Zona Dahi')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(zona['mata'], cv2.COLOR_BGR2RGB))
    plt.title('Zona Mata')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(zona['pipi'], cv2.COLOR_BGR2RGB))
    plt.title('Zona Pipi')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(zona['mulut'], cv2.COLOR_BGR2RGB))
    plt.title('Zona Mulut')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/demo_segmentasi.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print informasi
    print("=== Informasi Segmentasi ===")
    print(f"Gambar diproses: {image_path}")
    print("\nUkuran setiap zona:")
    for nama, zona_img in zona.items():
        print(f"{nama}: {zona_img.shape}")
    
    return zona

# Contoh penggunaan
if __name__ == "__main__":
    # Gunakan gambar dari dataset
    sample_image = 'dataset/high/1.jpg'
    
    if os.path.exists(sample_image):
        print("Memproses segmentasi wajah...")
        zona = tampilkan_hasil(sample_image)
        print("\n✅ Segmentasi selesai!")
        print("📊 Gambar disimpan di: output/demo_segmentasi.png")
    else:
        print("File tidak ditemukan")
