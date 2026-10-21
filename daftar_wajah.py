
from pathlib import Path
import json

# Folder utama foto mahasiswa
FOLDER_DATA = Path("data_wajah")

# Data identitas mahasiswa
MAHASISWA = {
    "A320250010": {
        "nama": "Gavra Darius Ulul Albab",
        "folder_foto": "A320250010",
    }
}

def simpan_daftar_mahasiswa():
    FOLDER_DATA.mkdir(parents=True, exist_ok=True)

    for nim, data in MAHASISWA.items():
        folder_foto = FOLDER_DATA / data["folder_foto"]

        if not folder_foto.exists():
            print(f"[PERINGATAN] Folder foto belum ada: {folder_foto}")
            continue

        file_foto = [
            p.name
            for p in folder_foto.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if not file_foto:
            print(f"[PERINGATAN] Belum ada foto untuk {nim}")
            continue

        print(f"Mahasiswa: {data['nama']}")
        print(f"NIM: {nim}")
        print(f"Jumlah foto: {len(file_foto)}")
        print(f"Foto: {', '.join(file_foto)}")
        print("-" * 40)

    file_json = FOLDER_DATA / "mahasiswa.json"
    file_json.write_text(
        json.dumps(MAHASISWA, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Daftar mahasiswa tersimpan di: {file_json}")

if __name__ == "__main__":
    simpan_daftar_mahasiswa()
