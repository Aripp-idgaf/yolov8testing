
import cv2
import json
import os
import numpy as np
from insightface.app import FaceAnalysis

# Threshold sementara untuk pengujian, belum final
THRESHOLD_UJI = 0.50
MINIMAL_SELISIH = 0.10

# Muat model pengenalan wajah
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# Baca data mahasiswa
with open("data_wajah/mahasiswa.json", "r", encoding="utf-8") as f:
    mahasiswa = json.load(f)

# Buat embedding referensi dari foto terdaftar
referensi = []

for nim, data in mahasiswa.items():
    folder = os.path.join("data_wajah", data["folder_foto"])

    for nama_file in os.listdir(folder):
        if not nama_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        path = os.path.join(folder, nama_file)
        gambar = cv2.imread(path)

        if gambar is None:
            continue

        wajah = app.get(gambar)

        if len(wajah) == 1:
            referensi.append({
                "nim": nim,
                "nama": data["nama"],
                "embedding": wajah[0].normed_embedding
            })

if not referensi:
    raise SystemExit("Tidak ada foto referensi yang berhasil diproses.")

print(f"Foto referensi berhasil dimuat: {len(referensi)}")
print("Tekan Q pada jendela kamera untuk keluar.")

# Buka kamera
kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    raise SystemExit("Kamera tidak bisa dibuka. Coba indeks kamera 1.")

while True:
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Gagal membaca gambar dari kamera.")
        break

    wajah_terdeteksi = app.get(frame)

    for wajah in wajah_terdeteksi:
        x1, y1, x2, y2 = map(int, wajah.bbox)
        embedding = wajah.normed_embedding

        # Cari skor terbaik untuk setiap identitas
        skor_identitas = {}

        for ref in referensi:
            skor = float(np.dot(embedding, ref["embedding"]))
            nim = ref["nim"]

            if nim not in skor_identitas or skor > skor_identitas[nim]["skor"]:
                skor_identitas[nim] = {
                    "nama": ref["nama"],
                    "skor": skor
                }

        urutan = sorted(
            skor_identitas.items(),
            key=lambda item: item[1]["skor"],
            reverse=True
        )

        terbaik_nim, terbaik = urutan[0]
        skor_tertinggi = terbaik["skor"]

        if len(urutan) >= 2:
            selisih = skor_tertinggi - urutan[1][1]["skor"]
        else:
            selisih = 0.0

        if skor_tertinggi < THRESHOLD_UJI:
            nama_tampil = "TIDAK DIKENALI"
            warna = (0, 0, 255)
        elif selisih < MINIMAL_SELISIH:
            nama_tampil = "PERLU VERIFIKASI"
            warna = (0, 165, 255)
        else:
            nama_tampil = terbaik["nama"]
            warna = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)

        teks = f"{nama_tampil} ({skor_tertinggi:.2f})"
        posisi_y = max(25, y1 - 10)

        cv2.putText(
            frame, teks, (x1, posisi_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2
        )

    cv2.imshow("Uji Face Recognition - Tekan Q untuk keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
