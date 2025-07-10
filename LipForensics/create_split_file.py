"""
Skrip untuk membuat file split JSON kustom dari direktori data.
Ini akan membuat file JSON yang berisi daftar video asli dan palsu,
yang kemudian bisa digunakan oleh skrip train.py.

Run:
python create_split_file.py 
    --real_dir data/datasets/Forensics/RealFF/c23/cropped_mouths/ \
    --fake_dir data/datasets/Forensics/Face2Face/c23/cropped_mouths/
    --output_file data/splits/train_custom.json
"""
import os
import json
import argparse

def create_split_file(real_dir, fake_dir, output_file):
    """
    Membaca nama folder video dari direktori real dan fake,
    lalu menyimpannya ke dalam file JSON.
    """
    print(f"Membaca video asli dari: {real_dir}")
    try:
        # Dapatkan nama folder video (tanpa path lengkap)
        real_videos = [d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))]
        print(f"Ditemukan {len(real_videos)} video asli.")
    except FileNotFoundError:
        print(f"ERROR: Direktori tidak ditemukan: {real_dir}")
        return

    print(f"Membaca video palsu dari: {fake_dir}")
    try:
        fake_videos = [d for d in os.listdir(fake_dir) if os.path.isdir(os.path.join(fake_dir, d))]
        print(f"Ditemukan {len(fake_videos)} video palsu.")
    except FileNotFoundError:
        print(f"ERROR: Direktori tidak ditemukan: {fake_dir}")
        return
        
    # Format data sesuai yang diharapkan oleh skrip (list berisi dua list)
    # Contoh: [["000", "001", ...], ["000_001", "002_003", ...]]
    split_data = [real_videos, fake_videos]

    # Tulis data ke file JSON
    with open(output_file, 'w') as f:
        json.dump(split_data, f)
        
    print(f"\nFile split kustom berhasil dibuat di: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Membuat file split JSON kustom.")
    parser.add_argument('--real_dir', type=str, required=True, help='Path ke direktori berisi folder-folder video ASLI yang sudah diproses.')
    parser.add_argument('--fake_dir', type=str, required=True, help='Path ke direktori berisi folder-folder video PALSU yang sudah diproses.')
    parser.add_argument('--output_file', type=str, required=True, help='Path untuk menyimpan file .json yang dihasilkan.')
    args = parser.parse_args()

    create_split_file(args.real_dir, args.fake_dir, args.output_file)