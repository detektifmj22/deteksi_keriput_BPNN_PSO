from preprocessing import load_and_preprocess_images
from ekstraksi_fitur import ekstrak_fitur_keriput
from train_bpnn import load_model
import pandas as pd

# 1. Load gambar dan preprocessing
images, filenames = load_and_preprocess_images('dataset/high/')

# 2. Ekstrak fitur dari area wajah
fitur_data = [ekstrak_fitur_keriput(img) for img in images]

# 3. Load model BPNN
model = load_model('model/bpnn_model.pkl')

# 4. Prediksi distribusi keriput
prediksi = model.predict(fitur_data)

# 5. Simpan hasil
df_hasil = pd.DataFrame(fitur_data, columns=["dahi", "mata", "pipi", "mulut", "jumlah_kontur", "panjang_total_kontur"])
df_hasil['prediksi_keriput'] = prediksi
df_hasil['nama_file'] = filenames
df_hasil.to_csv('output/hasil_prediksi.csv', index=False)
print("Hasil prediksi disimpan di output/hasil_prediksi.csv")
