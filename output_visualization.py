import os
import cv2
import numpy as np

def preprocess_and_segment(image_path, output_dir):
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "1_grayscale.png"), gray)
    resized = cv2.resize(gray, (200, 200))
    cv2.imwrite(os.path.join(output_dir, "2_resized.png"), resized)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            print("Tidak ditemukan wajah pada gambar.")
            return
        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w = resized.shape[:2]

        # Indeks landmark untuk area dahi, pipi, mata, dan area mulut yang mencakup mulut dan dahi
        dahi_idx = [10, 338, 297, 332, 284, 389, 356, 454, 323]
        pipi_idx = [205, 50, 187, 80, 81, 82, 13, 312]
        mata_idx = [33, 246, 161, 160, 159, 158, 157, 173, 7, 163]
        mulut_idx = [61, 146, 91, 181, 84, 17, 314, 405, 10, 338, 297]

        def get_roi(idx_list):
            points = np.array([(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in idx_list])
            x, y, w_box, h_box = cv2.boundingRect(points)
            margin = 10
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w_box = min(w_box + 2 * margin, w - x)
            h_box = min(h_box + 2 * margin, h - y)
            roi = resized[y:y+h_box, x:x+w_box]
            if roi.size == 0:
                return None
            return roi

        dahi = get_roi(dahi_idx)
        pipi = get_roi(pipi_idx)
        mata = get_roi(mata_idx)
        mulut = get_roi(mulut_idx)

        if dahi is not None:
            cv2.imwrite(os.path.join(output_dir, "3_dahi_area.png"), dahi)
        if pipi is not None:
            cv2.imwrite(os.path.join(output_dir, "4_pipi_area.png"), pipi)
        if mata is not None:
            cv2.imwrite(os.path.join(output_dir, "5_mata_area.png"), mata)
        if mulut is not None:
            cv2.imwrite(os.path.join(output_dir, "6_mulut_area.png"), mulut)

    print(f"Output gambar pra-pemrosesan dan segmentasi disimpan di {output_dir}")

if __name__ == "__main__":
    IMAGE_FOLDER = "dataset/high"
    OUTPUT_DIR = "output/visualization_example"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_files[0])
        preprocess_and_segment(image_path, OUTPUT_DIR)
    else:
        print("Tidak ditemukan gambar di folder dataset/high")
