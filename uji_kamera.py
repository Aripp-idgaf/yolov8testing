import cv2
import time
import math
import os
import numpy as np
import psycopg2
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ================= 1. KONFIGURASI SISTEM =================
# Threshold dilonggarkan agar saat ganti ekspresi (senyum/mangap) tetap dikenali
THRESHOLD_UJI = 0.55
MINIMAL_SELISIH = 0.05

# Konfigurasi Geofencing (Radius Absensi)
KELAS_LAT = -6.9828        # Contoh latitude kelas
KELAS_LON = 110.4091       # Contoh longitude kelas
BATAS_RADIUS_METER = 50    # Maksimal jarak absen dari titik kelas (meter)

# MOCKUP GPS: Ubah angka ini untuk mengetes sistem menolak atau menerima jarakmu
GPS_MAHASISWA_SAAT_INI = (-6.9828, 110.4091)


# ================= 2. FUNGSI PEMBANTU =================
def hitung_jarak_meter(lat1, lon1, lat2, lon2):
    """Menghitung jarak dua titik koordinat bumi dalam meter (Rumus Haversine)"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def cek_wajah_asli_dasar(wajah_bgr):
    """Metode cadangan deteksi layar HP/kertas jika model ONNX belum ada"""
    abu_abu = cv2.cvtColor(wajah_bgr, cv2.COLOR_BGR2GRAY)
    tekstur = cv2.Laplacian(abu_abu, cv2.CV_64F).var()
    return 20 <= tekstur <= 1500


# ================= 3. PERSIAPAN DATABASE & AI =================
print("1. Menghubungkan ke PostgreSQL...")
try:
    koneksi = psycopg2.connect(
        host="10.137.197.64", database="postgres", user="postgres", password=os.environ["DB_PASSWORD"], port="5432"
    )
    kursor = koneksi.cursor()
    kursor.execute('''
        SELECT m.nim, m.nama, p.nama_prodi, m.vektor_wajah
        FROM mahasiswa m JOIN prodi p ON m.prodi_id = p.id_prodi
    ''')
    data_mahasiswa = kursor.fetchall()
    koneksi.close()
except Exception as e:
    raise SystemExit(f"Gagal terhubung ke database: {e}")

referensi = []
for baris in data_mahasiswa:
    referensi.append({
        "nim": baris[0], "nama": baris[1], "prodi": baris[2],
        "embedding": np.frombuffer(baris[3], dtype=np.float32)
    })
print(f"Berhasil memuat {len(referensi)} data.")

print("2. Memuat AI Pengenalan Wajah (InsightFace) di GPU...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

print("3. Memuat AI Anti-Spoofing (Passive Liveness)...")
FILE_MODEL_SPOOF = "antispoof.onnx"
pakai_ai_spoof = False
if os.path.exists(FILE_MODEL_SPOOF):
    try:
        # Muat file Anti-Spoofing ke dalam GPU
        spoof_ai = ort.InferenceSession(FILE_MODEL_SPOOF, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        pakai_ai_spoof = True
        print("[BERHASIL] AI Liveness (Anti-Spoofing) terdeteksi & aktif!")
    except Exception as e:
        print(f"[PERINGATAN] Gagal memuat antispoof.onnx: {e}")
else:
    print("[PERINGATAN] File 'antispoof.onnx' belum diunduh. Sistem memakai filter perlindungan dasar.")


# ================= 4. PROSES UTAMA (LOOP KAMERA) =================
# Hitung jarak sebelum kamera dibuka (karena ini masih GPS Statis/Mockup)
jarak_mahasiswa = hitung_jarak_meter(GPS_MAHASISWA_SAAT_INI[0], GPS_MAHASISWA_SAAT_INI[1], KELAS_LAT, KELAS_LON)

kamera = cv2.VideoCapture(0)
if not kamera.isOpened():
    raise SystemExit("Kamera tidak bisa dibuka.")

cv2.namedWindow("Sistem Absensi Final", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sistem Absensi Final", 800, 600)
prev_time = time.time()

while True:
    berhasil, frame = kamera.read()
    if not berhasil: break

    frame = cv2.flip(frame, 1)

    # ----------------------------------------------------
    # TAHAP 1: VALIDASI JARAK LOKASI
    # ----------------------------------------------------
    if jarak_mahasiswa > BATAS_RADIUS_METER:
        cv2.putText(frame, f"DITOLAK! Jarak Anda: {jarak_mahasiswa:.1f} Meter dari Kelas", (20, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "(Masuk ke area kelas 50m untuk absen)", (20, (frame.shape[0] // 2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        # ----------------------------------------------------
        # TAHAP 2: EKSEKUSI WAJAH GPU (LINEAR)
        # ----------------------------------------------------
        wajah_terdeteksi = app.get(frame)

        if len(wajah_terdeteksi) > 0:
            wajah_utama = max(wajah_terdeteksi, key=lambda w: (w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]))
            x1, y1, x2, y2 = map(int, wajah_utama.bbox)

            # Potong bagian wajah dari gambar keseluruhan
            wajah_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            is_asli = False

            # ----------------------------------------------------
            # TAHAP 3: PASSIVE LIVENESS / ANTI-SPOOFING
            # ----------------------------------------------------
            if wajah_crop.size > 0:
                if pakai_ai_spoof:
                    try:
                        # Siapkan format gambar untuk model MiniFASNet (80x80)
                        blob = cv2.dnn.blobFromImage(wajah_crop, 1.0/255.0, (80, 80), (0, 0, 0), swapRB=True, crop=False)
                        nama_input = spoof_ai.get_inputs()[0].name
                        hasil_spoof = spoof_ai.run(None, {nama_input: blob})[0]

                        kelas_terdeteksi = np.argmax(hasil_spoof)
                        skor_asli = hasil_spoof[0][1] # Asumsi index 1 adalah 'Wajah Nyata 3D'

                        is_asli = True if (kelas_terdeteksi == 1 and skor_asli > 0.80) else False
                    except:
                        is_asli = cek_wajah_asli_dasar(wajah_crop) # Mundur ke filter dasar jika struktur ONNX berbeda
                else:
                    is_asli = cek_wajah_asli_dasar(wajah_crop)

            # ----------------------------------------------------
            # TAHAP 4: PENCOCOKAN IDENTITAS (Hanya jika Lolos Spoof)
            # ----------------------------------------------------
            if not is_asli:
                nama_tampil = "LAYAR HP / PALSU"
                warna = (0, 0, 255)
                skor_tampil = 0.0
            else:
                embedding_kamera = wajah_utama.normed_embedding
                skor_identitas = {}

                for ref in referensi:
                    skor = float(np.dot(embedding_kamera, ref["embedding"]))
                    nim = ref["nim"]
                    if nim not in skor_identitas or skor > skor_identitas[nim]["skor"]:
                        skor_identitas[nim] = {"nama": ref["nama"], "skor": skor}

                urutan = sorted(skor_identitas.items(), key=lambda item: item[1]["skor"], reverse=True)

                if urutan:
                    _, terbaik = urutan[0]
                    skor_tertinggi = terbaik["skor"]

                    if skor_tertinggi < THRESHOLD_UJI:
                        nama_tampil = "TIDAK DIKENALI"
                        warna = (0, 0, 255)
                    else:
                        nama_tampil = f"{terbaik['nama']}"
                        warna = (0, 255, 0)
                    skor_tampil = skor_tertinggi
                else:
                    nama_tampil = "TIDAK DIKENALI"
                    warna = (0, 0, 255)
                    skor_tampil = 0.0

            # Gambar Kotak Bounding Box Akhir
            cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
            teks = f"{nama_tampil} ({skor_tampil:.2f})" if nama_tampil not in ["TIDAK DIKENALI", "LAYAR HP / PALSU"] else nama_tampil
            cv2.putText(frame, teks, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2)

    # ----------------------------------------------------
    # TAHAP 5: TAMPILAN DASHBOARD POJOK KIRI ATAS
    # ----------------------------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    teks_info = f"FPS: {fps:.1f} | Jarak: {jarak_mahasiswa:.1f}m | Anti-Spoof: {'AI (ONNX)' if pakai_ai_spoof else 'Dasar (Laplacian)'}"
    cv2.putText(frame, teks_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Sistem Absensi Final", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()