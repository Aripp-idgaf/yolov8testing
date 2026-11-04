import psycopg2
import cv2
import json
import os
import numpy as np
from insightface.app import FaceAnalysis

print("1. Menyiapkan Struktur Database PostgreSQL...")
koneksi = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="januar123", # <-- GANTI DENGAN PASSWORD-MU
    port="5432"
)
koneksi.autocommit = True
kursor = koneksi.cursor()

# Hapus tabel lama jika ingin mereset ulang strukturnya dengan aman
kursor.execute('DROP TABLE IF EXISTS mahasiswa CASCADE')
kursor.execute('DROP TABLE IF EXISTS prodi CASCADE')

# 1. Buat Tabel Prodi
kursor.execute('''
    CREATE TABLE prodi (
        id_prodi SERIAL PRIMARY KEY,
        nama_prodi VARCHAR UNIQUE NOT NULL
    )
''')

# 2. Buat Tabel Mahasiswa dengan Foreign Key ke Prodi
kursor.execute('''
    CREATE TABLE mahasiswa (
        nim VARCHAR PRIMARY KEY,
        nama VARCHAR NOT NULL,
        prodi_id INT REFERENCES prodi(id_prodi),
        vektor_wajah BYTEA NOT NULL
    )
''')

print("2. Memuat Mesin AI (InsightFace)...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(320, 320))

print("3. Memproses Data Mahasiswa & Prodi...\n")
with open("data_wajah/mahasiswa.json", "r", encoding="utf-8") as f:
    mahasiswa_data = json.load(f)

for nim, data in mahasiswa_data.items():
    nama = data["nama"]
    nama_prodi = data["prodi"]
    folder = os.path.join("data_wajah", data["folder_foto"])
    
    # Masukkan Prodi ke tabel prodi (jika sudah ada, abaikan menggunakan ON CONFLICT)
    kursor.execute(
        "INSERT INTO prodi (nama_prodi) VALUES (%s) ON CONFLICT (nama_prodi) DO NOTHING",
        (nama_prodi,)
    )
    
    # Ambil ID dari prodi tersebut
    kursor.execute("SELECT id_prodi FROM prodi WHERE nama_prodi = %s", (nama_prodi,))
    prodi_id = kursor.fetchone()[0]
    
    # Proses ekstraksi foto wajah
    list_embedding = []
    if os.path.exists(folder):
        for nama_file in os.listdir(folder):
            if not nama_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(folder, nama_file)
            gambar = cv2.imread(path)
            if gambar is None: continue
                
            wajah = app.get(gambar)
            if len(wajah) == 1:
                list_embedding.append(wajah[0].normed_embedding)
    
    if not list_embedding:
        print(f" [!] Gagal: Tidak ada wajah valid untuk {nama}")
        continue
        
    # Hitung vektor rata-rata & normalisasi
    vektor_gabungan = np.mean(list_embedding, axis=0)
    vektor_final = vektor_gabungan / np.linalg.norm(vektor_gabungan)
    vektor_biner = vektor_final.tobytes()
    
    # Simpan ke tabel mahasiswa beserta relasi prodi_id-nya
    kursor.execute(
        "INSERT INTO mahasiswa (nim, nama, prodi_id, vektor_wajah) VALUES (%s, %s, %s, %s)", 
        (nim, nama, prodi_id, vektor_biner)
    )
    print(f" [+] Berhasil mendaftarkan: {nama} ({nama_prodi})")

koneksi.close()
print("\nSelesai! Relasi tabel Prodi dan Mahasiswa berhasil dibuat di PostgreSQL.")