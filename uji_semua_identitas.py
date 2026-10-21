
import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Siapkan model pengenalan wajah
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# Baca data mahasiswa
with open("data_wajah/mahasiswa.json", "r", encoding="utf-8") as file:
    mahasiswa = json.load(file)


def ambil_embedding(path_foto):
    gambar = cv2.imread(path_foto)

    if gambar is None:
        print(f"Foto tidak bisa dibaca: {path_foto}")
        return None

    wajah = app.get(gambar)

    if len(wajah) == 0:
        print(f"Tidak ada wajah terdeteksi: {path_foto}")
        return None

    if len(wajah) > 1:
        print(f"PERINGATAN: Ada {len(wajah)} wajah di {path_foto}.")
        print("Gunakan foto dengan satu wajah saja.")
        return None

    return wajah[0].normed_embedding


# Masukkan lokasi foto yang ingin diuji
foto_uji = input(
    "Masukkan lokasi foto uji (contoh: foto_uji/gavra.jpg): "
).strip().strip('"')

embedding_uji = ambil_embedding(foto_uji)

if embedding_uji is None:
    print("Pengujian dibatalkan karena wajah tidak dapat diproses.")
    raise SystemExit

hasil = []

# Bandingkan foto uji dengan seluruh foto referensi
for nim, data in mahasiswa.items():
    folder = os.path.join("data_wajah", data["folder_foto"])
    nilai_identitas = []

    if not os.path.isdir(folder):
        print(f"Folder tidak ditemukan: {folder}")
        continue

    for nama_file in os.listdir(folder):
        if not nama_file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        path_referensi = os.path.join(folder, nama_file)
        embedding_referensi = ambil_embedding(path_referensi)

        if embedding_referensi is None:
            continue

        kemiripan = float(
            np.dot(embedding_uji, embedding_referensi)
        )

        nilai_identitas.append((nama_file, kemiripan))

    if nilai_identitas:
        skor_tertinggi = max(nilai_identitas, key=lambda x: x[1])

        hasil.append({
            "nim": nim,
            "nama": data["nama"],
            "skor": skor_tertinggi[1],
            "foto": skor_tertinggi[0],
            "semua_skor": nilai_identitas
        })


# Tampilkan hasil
hasil.sort(key=lambda x: x["skor"], reverse=True)

# AMBANG UJI SEMENTARA, BUKAN THRESHOLD FINAL
THRESHOLD_UJI = 0.50
MINIMAL_SELISIH = 0.10

print("\n===== HASIL PENGUJIAN WAJAH =====")

if not hasil:
    print("Tidak ada foto referensi yang berhasil diproses.")
else:
    for item in hasil:
        print(f"\nNama       : {item['nama']}")
        print(f"NIM        : {item['nim']}")
        print("Kemiripan per foto:")

        for nama_foto, skor in item["semua_skor"]:
            print(f"  - {nama_foto}: {skor:.4f}")

        print(f"Skor tertinggi identitas: {item['skor']:.4f}")

    terbaik = hasil[0]
    skor_tertinggi = terbaik["skor"]

    if len(hasil) >= 2:
        kandidat_kedua = hasil[1]
        selisih = skor_tertinggi - kandidat_kedua["skor"]
    else:
        kandidat_kedua = None
        selisih = 0.0

    print("\n===== KEPUTUSAN PENGENALAN SEMENTARA =====")
    print(f"Identitas teratas : {terbaik['nama']}")
    print(f"NIM               : {terbaik['nim']}")
    print(f"Skor tertinggi    : {skor_tertinggi:.4f}")

    if kandidat_kedua:
        print(f"Kandidat kedua    : {kandidat_kedua['nama']}")
        print(f"Skor kandidat 2   : {kandidat_kedua['skor']:.4f}")

    print(f"Selisih skor      : {selisih:.4f}")
    print(f"Threshold uji     : {THRESHOLD_UJI:.2f}")
    print(f"Minimal selisih   : {MINIMAL_SELISIH:.2f}")

    if skor_tertinggi < THRESHOLD_UJI:
        status = "TIDAK DIKENALI"
        alasan = "Skor tertinggi di bawah threshold uji."
    elif selisih < MINIMAL_SELISIH:
        status = "PERLU VERIFIKASI"
        alasan = "Selisih skor dua kandidat terlalu kecil."
    else:
        status = "COCOK (UJI SEMENTARA)"
        alasan = "Skor dan selisih memenuhi aturan uji sementara."

    print(f"\nSTATUS: {status}")
    print(f"Keterangan: {alasan}")
    print("\nCatatan: status ini hanya untuk evaluasi.")
    print("Belum ada presensi yang disimpan.")
    print("Threshold harus divalidasi dengan lebih banyak data.")
