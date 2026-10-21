
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Memuat model pengenalan wajah
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def ambil_embedding(lokasi_gambar):
    gambar = cv2.imread(lokasi_gambar)

    if gambar is None:
        print(f"Gambar tidak dapat dibaca: {lokasi_gambar}")
        return None

    wajah = app.get(gambar)

    print(f"\nFile: {lokasi_gambar}")
    print(f"Jumlah wajah terdeteksi: {len(wajah)}")

    if len(wajah) == 0:
        print("Tidak ada wajah yang terdeteksi.")
        return None

    for i, f in enumerate(wajah):
        x1, y1, x2, y2 = f.bbox.astype(int)
        print(
            f"Wajah {i + 1}: "
            f"kotak=({x1}, {y1}, {x2}, {y2}), "
            f"confidence={float(f.det_score):.3f}"
        )

    wajah_utama = max(
        wajah,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    print("Wajah terbesar berhasil dipilih.")
    return wajah_utama.normed_embedding


embedding1 = ambil_embedding("assets/wajah1.jpg")
embedding2 = ambil_embedding("assets/wajah2.jpg")

if embedding1 is not None and embedding2 is not None:
    kemiripan = float(np.dot(embedding1, embedding2))

    print("Cosine similarity:", round(kemiripan, 4))
else:
    print("Pengujian belum dapat dilakukan.")
