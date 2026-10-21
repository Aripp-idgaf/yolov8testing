
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# Folder foto mahasiswa
folder = Path("data_wajah/A320250010")
foto1 = folder / "wajah2.jpg"
foto2 = folder / "wajah5.jpg"

# Memuat model InsightFace
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def ambil_embedding(lokasi):
    gambar = cv2.imread(str(lokasi))

    if gambar is None:
        print(f"Foto tidak dapat dibaca: {lokasi}")
        return None

    wajah = app.get(gambar)

    print(f"\nFile: {lokasi}")
    print(f"Jumlah wajah terdeteksi: {len(wajah)}")

    if len(wajah) != 1:
        print("Pengujian membutuhkan tepat satu wajah.")
        return None

    
    wajah_utama = wajah[0]

    print("Koordinat wajah:", wajah_utama.bbox)
    print("Landmark wajah:", wajah_utama.kps)
    print(f"Confidence: {float(wajah_utama.det_score):.3f}")

    embedding = wajah_utama.normed_embedding

    print("Panjang embedding:", len(embedding))
    print("Norma embedding:", np.linalg.norm(embedding))
    print("5 angka pertama:", embedding[:5])

    return embedding



embedding1 = ambil_embedding(foto1)
embedding2 = ambil_embedding(foto2)

if embedding1 is not None and embedding2 is not None:
    kemiripan = float(np.dot(embedding1, embedding2))

    print("\n" + "=" * 40)
    print("HASIL UJI KEMIRIPAN WAJAH")
    print("=" * 40)
    print(f"Foto pertama : {foto1.name}")
    print(f"Foto kedua   : {foto2.name}")
    print(f"Cosine similarity: {kemiripan:.4f}")
    print("=" * 40)
    print("Nilai ini perlu diuji dengan foto orang")
    print("yang sama dan berbeda sebelum menentukan")
    print("ambang pengenalan untuk presensi.")
else:
    print("\nPerbandingan belum dapat dilakukan.")
