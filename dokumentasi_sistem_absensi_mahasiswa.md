# Dokumentasi Sistem Absensi Pengenalan Wajah Mahasiswa

Sistem absensi berbasis *Computer Vision* yang menggunakan **YOLOv11 / InsightFace** untuk ekstraksi fitur wajah, terintegrasi penuh dengan **PostgreSQL** dalam arsitektur *Client-Server*, dan diakselerasi menggunakan GPU (NVIDIA RTX 3050).

---

## 🌟 Fitur Utama
1. **Face Recognition Akurasi Tinggi:** Menggunakan model `buffalo_l` (InsightFace) dengan perhitungan *Cosine Similarity* (Ambang batas standar: 0.55).
2. **GPU Acceleration:** Pemrosesan AI dialihkan sepenuhnya ke GPU (`CUDAExecutionProvider`, `ctx_id=0`) untuk memastikan *real-time processing* di atas 20-30 FPS.
3. **Database Terpusat (Client-Server):** Penyimpanan vektor wajah, jadwal, dan riwayat absensi terpusat di server PostgreSQL yang bisa diakses via jaringan Wi-Fi lokal (Port 5432).
4. **Geofencing:** Radius absensi dibatasi (contoh: 50 meter) menggunakan rumus Haversine untuk memvalidasi lokasi GPS mahasiswa dengan koordinat kelas.
5. **Anti-Spoofing (Liveness Detection):** Jalur khusus terintegrasi untuk model `antispoof.onnx` (MiniFASNet) guna memblokir kecurangan menggunakan foto cetak atau layar HP.
6. **Smart Cooldown & Debouncing:** Mencegah antrean jaringan (*I/O Blocking*) dan *spam* database dengan memberikan jeda waktu (cooldown) pencatatan absensi antar frame kamera.

---

## 🛠️ Prasyarat Sistem
*   **Perangkat Keras:** Webcam, GPU NVIDIA (Mendukung CUDA, misal: RTX 3050).
*   **Perangkat Lunak:** Python 3.9+, PostgreSQL (Versi 15 atau 18), pgAdmin 4.
*   **Pustaka Python Utama:** `opencv-python`, `numpy`, `insightface`, `onnxruntime-gpu`, `psycopg2`, `python-dotenv`.

---

## 🚀 Panduan Konfigurasi Server (PostgreSQL)

Agar database PostgreSQL di laptop server dapat diakses oleh laptop lain di jaringan Wi-Fi yang sama:

1. **Buka Akses IP (`postgresql.conf`)**
   Lokasi: `C:\Program Files\PostgreSQL\<versi>\data\postgresql.conf`
   Ubah baris listen address menjadi:
   ```ini
   listen_addresses = '*'
   ```
2. **Izinkan Koneksi (`pg_hba.conf`)**
   Lokasi: `C:\Program Files\PostgreSQL\<versi>\data\pg_hba.conf`
   Tambahkan di baris paling bawah:
   ```text
   host    all             all             0.0.0.0/0               scram-sha-256
   ```
3. **Buka Port Windows Firewall**
   Buat *Inbound Rule* baru di **Windows Defender Firewall** untuk mengizinkan koneksi masuk ke Port TCP **5432**.
4. **Restart Layanan**
   Buka aplikasi **Services** di Windows, cari `postgresql-x64-<versi>`, klik kanan, lalu **Restart**.

---

## ⚙️ Langkah Instalasi & Penggunaan (Sisi Client)

**1. Persiapan Environment (`.env`)**
Buat file `.env` di folder utama aplikasi dan sesuaikan IP server (Host):
```env
DB_HOST=10.137.197.64  # Ganti dengan IPv4 Wi-Fi laptop Server
DB_PORT=5432
DB_NAME=db_project
DB_USER=postgres
DB_PASSWORD=password_kamu
```

**2. Pembuatan Struktur Tabel (Inisialisasi)**
Jika kamu menemui error `psycopg2.errors.UndefinedTable: relation "t_datawajah" does not exist`, tabel fisik belum dibuat. Jalankan perintah ini satu kali:
```bash
python -c "import database; print(database.create_tables())"
```

**3. Registrasi Wajah**
Tabel yang baru dibuat masih kosong. Daftarkan identitas dan wajah awal mahasiswa dengan menjalankan:
```bash
python registrasi_wajah.py
```
*(Model InsightFace `buffalo_l` akan diunduh otomatis pada tahap ini jika belum ada di direktori lokal).*

**4. Menjalankan Mesin Absensi Utama**
Jalankan modul deteksi kamera:
```bash
python uji_kamera_3.py
```

---

## 💡 Pemecahan Masalah (Troubleshooting)

| Masalah / Error | Penyebab | Solusi |
| :--- | :--- | :--- |
| **FPS Drop (1-2 FPS)** | GPU mati, AI dijalankan di CPU. | Pastikan `app.prepare(ctx_id=0)` di kode utama (jangan gunakan `-1`). |
| **FPS Drop (Saat Wajah Muncul)** | Operasi Anti-Spoofing berbasis `Laplacian` sangat berat bagi CPU. | Matikan/Bypass blok *Laplacian* sementara, atau gunakan model `antispoof.onnx`. |
| **Terminal Spam "JADWAL KOSONG"** | Kesalahan penempatan variabel update `cooldown timer`. | Pastikan *update timer* (`riwayat_absen_terakhir[nim] = waktu_sekarang`) dieksekusi **sebelum** pengecekan `id_jadwal_aktif`. |
| **FATAL: password authentication failed** | Kredensial di file `.env` salah, atau modul `dotenv` belum dimuat. | Pastikan `python-dotenv` terinstal dan file `.env` terbaca oleh modul `config.py`. |
| **DITOLAK! Jarak Anda...** | Sistem Geofencing mendeteksi koordinat GPS *mockup* berada di luar radius 50m. | Sesuaikan variabel `GPS_MOCK_LAT` dan `GPS_MOCK_LON` di file `config.py` dengan koordinat kelas sesungguhnya. |

---
*Dokumentasi ini dibuat untuk proyek rekayasa perangkat lunak sistem absensi pengenalan wajah.*