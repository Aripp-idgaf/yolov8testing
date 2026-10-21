import threading
import time
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
app.prepare(ctx_id=-1, det_size=(320, 320))

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



# Data bersama antara thread kamera, AI, dan tampilan
lock = threading.Lock()
berhenti = threading.Event()

frame_terbaru = None
nomor_frame = 0
hasil_terakhir = []


def ambil_kamera():
    global frame_terbaru, nomor_frame

    while not berhenti.is_set():
        berhasil, frame = kamera.read()

        if berhasil:
            with lock:
                frame_terbaru = frame
                nomor_frame += 1
        else:
            time.sleep(0.01)


def proses_ai():
    global hasil_terakhir

    frame_diproses = -1

    while not berhenti.is_set():
        with lock:
            if frame_terbaru is None or nomor_frame == frame_diproses:
                frame = None
            else:
                frame = frame_terbaru.copy()
                frame_diproses = nomor_frame

        if frame is None:
            time.sleep(0.005)
            continue

        # Batasi pengenalan sekitar 5 kali per detik
        waktu_mulai = time.time()
        wajah_terdeteksi = app.get(frame)
        hasil_baru = []

        for wajah in wajah_terdeteksi:
            x1, y1, x2, y2 = map(int, wajah.bbox)
            embedding = wajah.normed_embedding

            skor_identitas = {}

            for ref in referensi:
                skor = float(np.dot(embedding, ref["embedding"]))
                nim = ref["nim"]

                if (
                    nim not in skor_identitas
                    or skor > skor_identitas[nim]["skor"]
                ):
                    skor_identitas[nim] = {
                        "nama": ref["nama"],
                        "skor": skor
                    }

            urutan = sorted(
                skor_identitas.items(),
                key=lambda item: item[1]["skor"],
                reverse=True
            )

            if not urutan:
                continue

            _, terbaik = urutan[0]
            skor_tertinggi = terbaik["skor"]

            selisih = (
                skor_tertinggi - urutan[1][1]["skor"]
                if len(urutan) >= 2 else 0.0
            )

            if skor_tertinggi < THRESHOLD_UJI:
                nama_tampil = "TIDAK DIKENALI"
                warna = (0, 0, 255)
            elif selisih < MINIMAL_SELISIH:
                nama_tampil = "PERLU VERIFIKASI"
                warna = (0, 165, 255)
            else:
                nama_tampil = terbaik["nama"]
                warna = (0, 255, 0)

            hasil_baru.append({
                "bbox": (x1, y1, x2, y2),
                "nama": nama_tampil,
                "skor": skor_tertinggi,
                "warna": warna
            })

        with lock:
            hasil_terakhir = hasil_baru

        # Jeda tambahan agar inferensi tidak terus membebani CPU
        waktu_proses = time.time() - waktu_mulai
        if waktu_proses < 0.20:
            time.sleep(0.20 - waktu_proses)


# Jalankan pengambilan kamera dan AI secara terpisah
thread_kamera = threading.Thread(
    target=ambil_kamera,
    daemon=True
)
thread_ai = threading.Thread(
    target=proses_ai,
    daemon=True
)

thread_kamera.start()
thread_ai.start()

print("Kamera aktif. Tekan Q pada jendela kamera untuk keluar.")

try:
    while not berhenti.is_set():
        with lock:
            if frame_terbaru is None:
                frame_tampil = None
            else:
                frame_tampil = frame_terbaru.copy()

            hasil_tampil = list(hasil_terakhir)

        if frame_tampil is None:
            time.sleep(0.005)
            continue

        for hasil in hasil_tampil:
            x1, y1, x2, y2 = hasil["bbox"]
            warna = hasil["warna"]

            cv2.rectangle(
                frame_tampil,
                (x1, y1),
                (x2, y2),
                warna,
                2
            )

            teks = f"{hasil['nama']} ({hasil['skor']:.2f})"

            cv2.putText(
                frame_tampil,
                teks,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                warna,
                2
            )

        cv2.imshow(
            "Uji Face Recognition - Tekan Q untuk keluar",
            frame_tampil
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    berhenti.set()
    thread_kamera.join(timeout=2)
    thread_ai.join(timeout=2)
    kamera.release()
    cv2.destroyAllWindows()
