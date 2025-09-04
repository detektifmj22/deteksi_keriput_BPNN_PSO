import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def ekstrak_glcm_fitur(area):
    glcm = graycomatrix(area, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    return (contrast + homogeneity + energy) / 3

def ekstrak_fitur_keriput(img):
    dahi = img[20:60, 60:140]
    mata = img[70:100, 60:140]
    pipi = img[110:140, 50:150]
    mulut = img[150:180, 70:130]
    fitur_dahi = ekstrak_glcm_fitur(dahi)
    fitur_mata = ekstrak_glcm_fitur(mata)
    fitur_pipi = ekstrak_glcm_fitur(pipi)
    fitur_mulut = ekstrak_glcm_fitur(mulut)

    # Deteksi kontur pada gambar grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(img_blur, 30, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_kontur = len(contours)
    panjang_total_kontur = sum([cv2.arcLength(cnt, False) for cnt in contours])

    return [fitur_dahi, fitur_mata, fitur_pipi, fitur_mulut, jumlah_kontur, panjang_total_kontur]

from preprocessing import load_and_preprocess_images

if __name__ == "__main__":
    import os
    import pandas as pd

    data = []
    base_dir = "dataset"
    kelas_list = ["high", "medium"]  # Hanya kelas high dan medium, low dihilangkan
    for kelas in kelas_list:
        folder = os.path.join(base_dir, kelas)
        label = kelas
        images, filenames = load_and_preprocess_images(folder)
        for img, filename in zip(images, filenames):
            fitur = ekstrak_fitur_keriput(img)
            data.append({
                'dahi': fitur[0],
                'mata': fitur[1],
                'pipi': fitur[2],
                'mulut': fitur[3],
                'jumlah_kontur': fitur[4],
                'panjang_total_kontur': fitur[5],
                'label': label
            })

    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "keriput_dataset.csv"), index=False)
    print(f"Ekstraksi selesai. Data disimpan di {os.path.join(base_dir, 'keriput_dataset.csv')}")
