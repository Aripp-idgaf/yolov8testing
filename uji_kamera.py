import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis

# Threshold
# REVISI: Menaikkan threshold agar lebih ketat (menghindari False Positive)
THRESHOLD_UJI = 0.65
MINIMAL_SELISIH = 0.10
EAR_THRESHOLD = 0.20
MINIMUM_MATA_TERTUTUP_FRAMES = 2

# 1. TARIK DATA DARI DATABASE POSTGRESQL (DENGAN RELASI PRODI)
print("1. Menghubungkan ke PostgreSQL...")
try:
    koneksi = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="januar123", # Password sesuai file aslimu
        port="5432"
    )
    kursor = koneksi.cursor()
    
    # Mengambil data mahasiswa beserta nama prodinya menggunakan JOIN
    kursor.execute('''
        SELECT m.nim, m.nama, p.nama_prodi, m.vektor_wajah 
        FROM mahasiswa m
        JOIN prodi p ON m.prodi_id = p.id_prodi
    ''')
    data_mahasiswa = kursor.fetchall()
    koneksi.close()
except Exception as e:
    raise SystemExit(f"Gagal terhubung ke database: {e}")

referensi = []
for baris in data_mahasiswa:
    nim = baris[0]
    nama = baris[1]
    nama_prodi = baris[2]
    vektor_biner = baris[3]
    
    # Ubah data biner (BYTEA) kembali menjadi matriks 512 angka
    embedding = np.frombuffer(vektor_biner, dtype=np.float32)
    
    referensi.append({
        "nim": nim,
        "nama": nama,
        "prodi": nama_prodi,
        "embedding": embedding
    })

if not referensi:
    raise SystemExit("Database kosong! Belum ada wajah yang terdaftar.")

print(f"Berhasil memuat {len(referensi)} data mahasiswa beserta prodinya dari PostgreSQL.")

# 2. MUAT MESIN AI
print("2. Memuat AI InsightFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(320, 320))

# 3. BUKA KAMERA & THREADING
# 3. SIAPKAN LIVENESS DETECTION
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

MATA_KIRI = (362, 385, 387, 263, 373, 380)
MATA_KANAN = (33, 160, 158, 133, 153, 144)


def hitung_ear(titik_wajah, indeks_mata, lebar, tinggi):
    titik = np.array(
        [
            (titik_wajah.landmark[i].x * lebar, titik_wajah.landmark[i].y * tinggi)
            for i in indeks_mata
        ],
        dtype=np.float32,
    )
    jarak_vertikal_1 = np.linalg.norm(titik[1] - titik[5])
    jarak_vertikal_2 = np.linalg.norm(titik[2] - titik[4])
    jarak_horizontal = np.linalg.norm(titik[0] - titik[3])
    if jarak_horizontal == 0:
        return 0.0
    return float((jarak_vertikal_1 + jarak_vertikal_2) / (2.0 * jarak_horizontal))


def hitung_iou(kotak_a, kotak_b):
    x1 = max(kotak_a[0], kotak_b[0])
    y1 = max(kotak_a[1], kotak_b[1])
    x2 = min(kotak_a[2], kotak_b[2])
    y2 = min(kotak_a[3], kotak_b[3])
    luas_irisan = max(0, x2 - x1) * max(0, y2 - y1)
    luas_a = max(0, kotak_a[2] - kotak_a[0]) * max(0, kotak_a[3] - kotak_a[1])
    luas_b = max(0, kotak_b[2] - kotak_b[0]) * max(0, kotak_b[3] - kotak_b[1])
    luas_gabungan = luas_a + luas_b - luas_irisan
    return luas_irisan / luas_gabungan if luas_gabungan > 0 else 0.0


def cari_landmark_wajah_utama(hasil_mesh, bbox_wajah, lebar, tinggi):
    kandidat_terbaik = None
    iou_terbaik = 0.0
    for landmark_wajah in hasil_mesh.multi_face_landmarks or []:
        x = [titik.x * lebar for titik in landmark_wajah.landmark]
        y = [titik.y * tinggi for titik in landmark_wajah.landmark]
        bbox_mesh = (min(x), min(y), max(x), max(y))
        iou = hitung_iou(bbox_wajah, bbox_mesh)
        if iou > iou_terbaik:
            iou_terbaik = iou
            kandidat_terbaik = landmark_wajah
    return kandidat_terbaik if iou_terbaik > 0 else None


# 4. BUKA KAMERA & THREADING
kamera = cv2.VideoCapture(0)
if not kamera.isOpened():
    face_mesh.close()
    raise SystemExit("Kamera tidak bisa dibuka.")

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
            frame = cv2.flip(frame, 1) 
            with lock:
                frame_terbaru = frame
                nomor_frame += 1
        else:
            time.sleep(0.01)

def proses_ai():
    global hasil_terakhir
    frame_diproses = -1
    liveness_aktif = False
    mata_tertutup_frames = 0
    bbox_wajah_terakhir = None

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

        tinggi_frame, lebar_frame = frame.shape[:2]
        wajah_terdeteksi = app.get(frame)
        hasil_baru = []

        # Hanya proses 1 wajah terbesar jika ada wajah yang terdeteksi.
        if len(wajah_terdeteksi) > 0:
            wajah_utama = max(wajah_terdeteksi, key=lambda w: (w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]))
            x1, y1, x2, y2 = map(int, wajah_utama.bbox)
            bbox_wajah = (x1, y1, x2, y2)

            if bbox_wajah_terakhir is not None and hitung_iou(
                bbox_wajah_terakhir, bbox_wajah
            ) < 0.10:
                liveness_aktif = False
                mata_tertutup_frames = 0
            bbox_wajah_terakhir = bbox_wajah

            ear = None
            if not liveness_aktif:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hasil_mesh = face_mesh.process(frame_rgb)
                titik_wajah = cari_landmark_wajah_utama(
                    hasil_mesh, bbox_wajah, lebar_frame, tinggi_frame
                )

                if titik_wajah is not None:
                    ear_kiri = hitung_ear(
                        titik_wajah, MATA_KIRI, lebar_frame, tinggi_frame
                    )
                    ear_kanan = hitung_ear(
                        titik_wajah, MATA_KANAN, lebar_frame, tinggi_frame
                    )
                    ear = (ear_kiri + ear_kanan) / 2.0

                    if ear < EAR_THRESHOLD:
                        mata_tertutup_frames += 1
                    else:
                        if mata_tertutup_frames >= MINIMUM_MATA_TERTUTUP_FRAMES:
                            liveness_aktif = True
                        mata_tertutup_frames = 0
                else:
                    mata_tertutup_frames = 0

            if not liveness_aktif:
                hasil_baru.append(
                    {
                        "bbox": bbox_wajah,
                        "nama": "Silakan Berkedip untuk Absen!",
                        "skor": None,
                        "warna": (0, 165, 255),
                        "liveness": False,
                        "ear": ear,
                    }
                )
            else:
                # Tidak ada perbandingan embedding atau skor database sebelum blink.
                embedding_kamera = wajah_utama.normed_embedding
                skor_identitas = {}
                for ref in referensi:
                    skor = float(np.dot(embedding_kamera, ref["embedding"]))
                    nim = ref["nim"]

                    if nim not in skor_identitas or skor > skor_identitas[nim]["skor"]:
                        skor_identitas[nim] = {
                            "nama": ref["nama"],
                            "prodi": ref["prodi"],
                            "skor": skor,
                        }

                urutan = sorted(
                    skor_identitas.items(),
                    key=lambda item: item[1]["skor"],
                    reverse=True,
                )

                if urutan:
                    _, terbaik = urutan[0]
                    skor_tertinggi = terbaik["skor"]
                    selisih = (
                        skor_tertinggi - urutan[1][1]["skor"]
                        if len(urutan) >= 2
                        else 0.0
                    )

                    if skor_tertinggi < THRESHOLD_UJI:
                        nama_tampil = "TIDAK DIKENALI"
                        warna = (0, 0, 255)
                    elif selisih < MINIMAL_SELISIH:
                        nama_tampil = "PERLU VERIFIKASI"
                        warna = (0, 165, 255)
                    else:
                        nama_tampil = f"{terbaik['nama']} - Absen Berhasil!"
                        warna = (0, 255, 0)

                    hasil_baru.append(
                        {
                            "bbox": bbox_wajah,
                            "nama": nama_tampil,
                            "skor": skor_tertinggi,
                            "warna": warna,
                            "liveness": True,
                            "ear": ear,
                        }
                    )
        else:
            liveness_aktif = False
            mata_tertutup_frames = 0
            bbox_wajah_terakhir = None

        with lock:
            hasil_terakhir = hasil_baru

thread_kamera = threading.Thread(target=ambil_kamera, daemon=True)
thread_ai = threading.Thread(target=proses_ai, daemon=True)
thread_kamera.start()
thread_ai.start()

print("Kamera aktif. Tekan Q pada jendela kamera untuk keluar.")
prev_time = time.time()

cv2.namedWindow("Uji Absensi dengan Relasi Prodi", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Uji Absensi dengan Relasi Prodi", 800, 600)

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

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame_tampil, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for hasil in hasil_tampil:
            x1, y1, x2, y2 = hasil["bbox"]
            warna = hasil["warna"]
            cv2.rectangle(frame_tampil, (x1, y1), (x2, y2), warna, 2)
            if hasil["skor"] is None:
                teks = hasil["nama"]
            else:
                teks = f"{hasil['nama']} ({hasil['skor']:.2f})"
            cv2.putText(frame_tampil, teks, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2)
            status_liveness = f"Liveness = {hasil['liveness']}"
            cv2.putText(
                frame_tampil,
                status_liveness,
                (x1, min(frame_tampil.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                warna,
                2,
            )

        cv2.imshow("Uji Absensi dengan Relasi Prodi", frame_tampil)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    berhenti.set()
    thread_kamera.join(timeout=2)
    thread_ai.join(timeout=2)
    kamera.release()
    face_mesh.close()
    cv2.destroyAllWindows()