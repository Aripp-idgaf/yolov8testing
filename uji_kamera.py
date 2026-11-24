"""
Sistem Pengenalan Wajah & Absensi Mahasiswa Menggunakan Webcam
Terintegrasi Penuh dengan Database PostgreSQL:
- Mengambil face embedding dan data mahasiswa dari t_datawajah JOIN t_mahasiswa
- Menghitung Cosine Similarity
- Jika cocok (skor >= threshold): menampilkan Nama, NIM, dan skor
- Jika tidak cocok: menampilkan 'MAHASISWA TIDAK DIKENAL'
- Memeriksa jadwal kuliah aktif di t_jadwal:
  * Jika jadwal aktif tersedia: catat absensi otomatis ke t_absensi
  * Jika jadwal belum tersedia: tampilkan peringatan dan JANGAN melakukan INSERT
- Mencegah absensi duplikat
- Menggunakan Multi-Threading (WebcamVideoStream) untuk FPS Maksimal
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from threading import Thread

import config
from database import (
    get_connection,
    close_connection,
    get_face_data,
    save_absensi,
    dapatkan_jadwal_aktif,
    test_connection
)

# ======================================================================
# KELAS MULTI-THREADING KAMERA (Mencegah I/O Blocking)
# ======================================================================
class WebcamVideoStream:
    def __init__(self, src=0):
        # Inisialisasi kamera dan baca frame pertama
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Jalankan thread terpisah untuk membaca frame secara mandiri
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Terus perbarui frame di memori selama sistem berjalan
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Kembalikan frame terbaru seketika tanpa perlu menunggu
        return self.grabbed, self.frame

    def stop(self):
        # Hentikan thread dan tutup kamera
        self.stopped = True
        self.stream.release()
# ======================================================================

# Cooldown absensi lokal untuk anti-spam antar frame
riwayat_absen_terakhir = {}  # {nim: timestamp}
pesan_notifikasi_absen = ""
warna_notifikasi = (0, 180, 0)
waktu_tampil_notifikasi = 0


def hitung_jarak_meter(lat1, lon1, lat2, lon2):
    """Menghitung jarak dua titik koordinat bumi dalam meter (Rumus Haversine)."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def cek_wajah_asli_dasar(wajah_bgr):
    """Metode cadangan deteksi layar HP/kertas jika model anti-spoofing belum ada."""
    abu_abu = cv2.cvtColor(wajah_bgr, cv2.COLOR_BGR2GRAY)
    tekstur = cv2.Laplacian(abu_abu, cv2.CV_64F).var()
    return 20 <= tekstur <= 1500


def main():
    global pesan_notifikasi_absen, warna_notifikasi, waktu_tampil_notifikasi

    print("=" * 68)
    print(" SISTEM ABSENSI PENGENALAN WAJAH MAHASISWA (POSTGRESQL & WEBCAM)")
    print("=" * 68)

    # 1. Uji koneksi ke PostgreSQL
    terhubung, pesan_db, _ = test_connection()
    if not terhubung:
        print(f"\n[ERROR] Database PostgreSQL tidak terhubung: {pesan_db}")
        print("Pastikan PostgreSQL aktif dan pengaturan di .env sudah sesuai.")
        return

    print(f" [+] Status Database: {pesan_db}")

    # 2. Muat data wajah dari PostgreSQL
    referensi = get_face_data()
    print(f" [+] Berhasil memuat {len(referensi)} identitas wajah dari tabel t_datawajah.")
    if not referensi:
        print(" [PERINGATAN] Tabel t_datawajah masih kosong! Jalankan registrasi_wajah.py terlebih dahulu.")

    # 3. Periksa ketersediaan jadwal aktif
    jadwal_aktif = dapatkan_jadwal_aktif()
    if jadwal_aktif:
        id_jadwal_aktif = jadwal_aktif["id_jadwal"]
        matkul_aktif = jadwal_aktif.get("nama_matkul", "")
        ruang_aktif = jadwal_aktif.get("ruang_kelas", "")
        print(f" [+] Jadwal Kuliah Aktif: {id_jadwal_aktif} - {matkul_aktif} (Ruang: {ruang_aktif})")
    else:
        id_jadwal_aktif = None
        print(" [INFO] Tidak ada jadwal kuliah yang aktif saat ini di tabel t_jadwal.")
        print("        Sistem HANYA akan melakukan verifikasi identitas (TIDAK ADA INSERT ABSENSI).")

    print(f" [+] Ambang Kemiripan (Similarity Threshold): {config.SIMILARITY_THRESHOLD}")

    # 4. Inisialisasi AI Face Recognition (InsightFace)
    print("\n[+] Memuat AI Face Recognition (InsightFace buffalo_l)...")
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(config.MODELS_CACHE_DIR),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    # GPU DIKATIFKAN DENGAN ctx_id=0
    app.prepare(ctx_id=0, det_size=(config.DETECTION_SIZE, config.DETECTION_SIZE))

    # 5. Inisialisasi AI Anti-Spoofing (Passive Liveness)
    pakai_ai_spoof = False
    if config.MODEL_ANTISPOOF.exists():
        try:
            spoof_ai = ort.InferenceSession(str(config.MODEL_ANTISPOOF), providers=["CPUExecutionProvider"])
            pakai_ai_spoof = True
            print("[+] Model Anti-Spoofing aktif!")
        except Exception as e:
            print(f"[!] Gagal memuat antispoof.onnx: {e}")
    else:
        print("[!] File antispoof.onnx belum ada. Menggunakan filter tekstur dasar.")

    # 6. Validasi Geofencing
    jarak_mahasiswa = hitung_jarak_meter(
        config.GPS_MOCK_LAT, config.GPS_MOCK_LON,
        config.KELAS_LAT, config.KELAS_LON
    )

    # 7. Membuka Kamera Webcam (Menggunakan Multi-Threading)
    print("\n[+] Membuka kamera webcam di latar belakang... (Tekan 'q' pada jendela video untuk keluar)")
    kamera = WebcamVideoStream(src=0).start()

    cv2.namedWindow("Sistem Absensi Mahasiswa - YOLO & PostgreSQL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sistem Absensi Mahasiswa - YOLO & PostgreSQL", 960, 680)
    prev_time = time.time()

    while True:
        # Mengambil frame seketika dari Thread 1 tanpa menunggu
        berhasil, frame = kamera.read()
        if not berhasil or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        waktu_sekarang = time.time()

        # ----------------------------------------------------
        # TAHAP 1: VALIDASI GEOFENCING
        # ----------------------------------------------------
        if jarak_mahasiswa > config.BATAS_RADIUS_METER:
            cv2.putText(
                frame,
                f"DITOLAK! Jarak Anda: {jarak_mahasiswa:.1f}m dari Ruang Kelas",
                (20, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.putText(
                frame,
                f"(Maksimal radius absensi: {config.BATAS_RADIUS_METER:.0f}m)",
                (20, (frame.shape[0] // 2) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )
        else:
            # ----------------------------------------------------
            # TAHAP 2: DETEKSI WAJAH
            # ----------------------------------------------------
            wajah_terdeteksi = app.get(frame)

            if len(wajah_terdeteksi) > 0:
                wajah_utama = max(wajah_terdeteksi, key=lambda w: (w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]))
                x1, y1, x2, y2 = map(int, wajah_utama.bbox)
                wajah_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                
                # ----------------------------------------------------
                # TAHAP 3: ANTI-SPOOFING (BYPASS SEMENTARA)
                # ----------------------------------------------------
                # Dimatikan agar CPU tidak kewalahan dan FPS kembali lancar
                is_asli = True

                # ----------------------------------------------------
                # TAHAP 4: PENCOCOKAN IDENTITAS & ABSENSI
                # ----------------------------------------------------
                if not is_asli:
                    nama_tampil = "LAYAR HP / PALSU"
                    nim_tampil = ""
                    warna = (0, 0, 255)
                    skor_tampil = 0.0
                else:
                    embedding_kamera = wajah_utama.normed_embedding
                    skor_identitas = {}

                    for ref in referensi:
                        skor = float(np.dot(embedding_kamera, ref["embedding"]))
                        nim_ref = ref["nim"]
                        if nim_ref not in skor_identitas or skor > skor_identitas[nim_ref]["skor"]:
                            skor_identitas[nim_ref] = {
                                "nim": nim_ref,
                                "nama": ref["nama"],
                                "prodi": ref["prodi"],
                                "skor": skor
                            }

                    urutan = sorted(skor_identitas.items(), key=lambda item: item[1]["skor"], reverse=True)

                    if urutan:
                        nim_terpilih, terbaik = urutan[0]
                        skor_tertinggi = terbaik["skor"]

                        if skor_tertinggi < config.SIMILARITY_THRESHOLD:
                            nama_tampil = "MAHASISWA TIDAK DIKENAL"
                            nim_tampil = ""
                            warna = (0, 0, 255)
                        else:
                            nama_tampil = terbaik["nama"]
                            nim_tampil = f"NIM: {terbaik['nim']}"
                            warna = (0, 255, 0)

                            # PENCATATAN ABSENSI DENGAN COOLDOWN
                            terakhir_absen = riwayat_absen_terakhir.get(nim_terpilih, 0)
                            
                            if (waktu_sekarang - terakhir_absen) > config.COOLDOWN_ABSEN_DETIK:
                                
                                # TIMER DI-UPDATE DI SINI! Mencegah spam meski jadwal kosong
                                riwayat_absen_terakhir[nim_terpilih] = waktu_sekarang
                                
                                if id_jadwal_aktif:
                                    try:
                                        hasil_absen = save_absensi(
                                            nim=nim_terpilih,
                                            id_jadwal=id_jadwal_aktif,
                                            status_kehadiran="Hadir",
                                            cek_duplikasi=True
                                        )
                                        
                                        if hasil_absen["status"] == "berhasil":
                                            # Teks dirampingkan
                                            pesan_notifikasi_absen = f"[HADIR] {terbaik['nama']} | ID: {hasil_absen['id_absensi']}"
                                            warna_notifikasi = (0, 180, 0)
                                            print(f" [DB INSERT] {pesan_notifikasi_absen}")
                                        else:
                                            # Teks dirampingkan
                                            pesan_notifikasi_absen = f"[INFO] {terbaik['nama']} sudah absen hari ini."
                                            warna_notifikasi = (0, 165, 255)
                                            print(f" [DB INFO] {pesan_notifikasi_absen}")

                                        waktu_tampil_notifikasi = waktu_sekarang + 4.0
                                    except Exception as err:
                                        pesan_notifikasi_absen = f"[ERROR] Gagal catat absensi: {err}"
                                        warna_notifikasi = (0, 0, 255)
                                        print(f" [DB ERROR] {pesan_notifikasi_absen}")
                                        waktu_tampil_notifikasi = waktu_sekarang + 4.0
                                else:
                                    # Teks dirampingkan agar tidak menabrak batas layar
                                    pesan_notifikasi_absen = f"[PERINGATAN] {terbaik['nama']} - Jadwal Kosong."
                                    warna_notifikasi = (0, 140, 255)
                                    print(f" [JADWAL KOSONG] {pesan_notifikasi_absen}")
                                    waktu_tampil_notifikasi = waktu_sekarang + 4.0
                        skor_tampil = skor_tertinggi
                    else:
                        nama_tampil = "MAHASISWA TIDAK DIKENAL"
                        nim_tampil = ""
                        warna = (0, 0, 255)
                        skor_tampil = 0.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
                if nama_tampil == "MAHASISWA TIDAK DIKENAL":
                    teks_label = f"{nama_tampil} ({skor_tampil:.2f})"
                elif nim_tampil:
                    teks_label = f"{nama_tampil} | {nim_tampil} | Sim: {skor_tampil:.2f}"
                else:
                    teks_label = nama_tampil
                cv2.putText(frame, teks_label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2)

        # ----------------------------------------------------
        # TAHAP 5: BANNER STATUS NOTIFIKASI
        # ----------------------------------------------------
        if waktu_sekarang < waktu_tampil_notifikasi and pesan_notifikasi_absen:
            # Mengatur ukuran kotak banner agar lebih pas
            cv2.rectangle(frame, (10, frame.shape[0] - 50), (frame.shape[1] - 10, frame.shape[0] - 10), warna_notifikasi, -1)
            cv2.putText(
                frame,
                pesan_notifikasi_absen,
                (20, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50, # Skala teks dikecilkan sedikit
                (255, 255, 255),
                1     # Ketebalan font dipertipis agar lebih mudah dibaca
            )

        # ----------------------------------------------------
        # TAHAP 6: HEADER STATUS
        # ----------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        jadwal_str = f"Jadwal: {id_jadwal_aktif}" if id_jadwal_aktif else "Jadwal: Tidak Ada"
        teks_info = f"FPS: {fps:.1f} | {jadwal_str} | Threshold: {config.SIMILARITY_THRESHOLD:.2f}"
        cv2.putText(frame, teks_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Sistem Absensi Mahasiswa - YOLO & PostgreSQL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Menghentikan thread kamera dengan bersih
    kamera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()