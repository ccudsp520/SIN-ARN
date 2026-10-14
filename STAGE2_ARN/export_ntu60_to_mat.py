"""Ekspor satu sampel NTU RGB+D 60 beserta label aksinya ke MATLAB.

Simpan file ini di folder utama hong_MSG3D, lalu jalankan di PyCharm.
Input yang didukung adalah data joint yang telah diproses, bukan file raw
.skeleton. Susunan array input: (N, 3, T, 25, P), dengan P = 1 atau 2.
Output X: (3, T, 25, P), sesuai visualize_ntu_skeleton.m.
Label dibaca dari pasangan (sample_names, labels) di val_label.pkl.
Nama aksi mengikuti kelas resmi NTU60:
https://github.com/shahroudy/NTURGB-D#action-classes

Koordinat, urutan joint, frame, dan kedua orang dipertahankan. Skrip ini
tidak menambahkan masking atau menjalankan model SIN.
"""

from pathlib import Path
import pickle
import re

import numpy as np
from scipy.io import savemat


PROJECT_DIR = Path(__file__).resolve().parent

# Lokasi default dataset NTU60 X-Sub di project hong_MSG3D.
# Jika lokasi dataset berbeda, ubah DATA_FILE menjadi path file yang benar.
DATA_FILE = PROJECT_DIR / "data" / "xsub" / "val_data_joint.npy"
LABEL_FILE = PROJECT_DIR / "data" / "xsub" / "val_label.pkl"

# Indeks Python: 0 berarti sampel pertama.
SAMPLE_INDEX = 30

# File ini akan dibuat di folder yang sama dengan skrip ini.
OUTPUT_FILE = PROJECT_DIR / "ntu60_sample30.mat"

# Urutan kelas A001 sampai A060. Label .pkl memakai indeks 0 sampai 59.
ACTION_NAMES = (
    "drink water", "eat meal/snack", "brush teeth", "brush hair",
    "drop object", "pick up object", "throw object", "sit down",
    "stand up", "clap", "read", "write", "tear paper", "put on jacket",
    "take off jacket", "put on shoe", "take off shoe", "put on glasses",
    "take off glasses", "put on hat/cap", "take off hat/cap", "cheer up",
    "wave hand", "kick object", "reach into pocket", "hop", "jump up",
    "phone call", "use phone/tablet", "type", "point", "take selfie",
    "check watch", "rub hands", "nod/bow", "shake head", "wipe face",
    "salute", "put palms together", "cross hands (stop)", "sneeze/cough",
    "stagger", "fall", "touch head", "touch chest", "touch back",
    "touch neck", "nausea/vomit", "fan oneself", "punch/slap another person",
    "kick another person", "push another person", "pat another person's back",
    "point at another person", "hug another person", "give something",
    "touch another person's pocket", "handshake", "walk towards each other",
    "walk apart from each other",
)


def read_action_metadata(label_file, sample_index, sample_count):
    """Baca label asli dan cocokkan dengan kode aksi pada nama rekaman."""
    label_file = Path(label_file).expanduser().resolve()
    if not label_file.is_file():
        raise FileNotFoundError(f"File label tidak ditemukan: {label_file}")
    with label_file.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("File label harus berisi pasangan (sample_names, labels).")
    names, labels = payload
    if len(names) != sample_count or len(labels) != sample_count:
        raise ValueError(
            "Jumlah nama dan label harus sama dengan jumlah sampel dataset. "
            "Pastikan DATA_FILE dan LABEL_FILE berasal dari split yang sama."
        )
    label = labels[sample_index]
    if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)):
        raise ValueError("Label NTU60 harus berupa integer 0 sampai 59.")
    label = int(label)
    if not 0 <= label < len(ACTION_NAMES):
        raise ValueError(f"Label {label} di luar kelas NTU60 (0 sampai 59).")
    raw_name = names[sample_index]
    sample_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    match = re.search(r"A(\d{3})(?=\.|$)", sample_name)
    if match and int(match.group(1)) != label + 1:
        raise ValueError(
            f"Nama rekaman {sample_name} tidak cocok dengan label {label}. "
            "Periksa pasangan file data dan label."
        )
    return {
        "sample_name": sample_name,
        "action_label": np.int32(label),
        "action_id": np.int32(label + 1),
        "action_code": f"A{label + 1:03d}",
        "action_name": ACTION_NAMES[label],
    }


def export_sample(data_file, sample_index, output_file, label_file=None):
    """Simpan satu sampel dan labelnya tanpa mengubah koordinatnya."""
    data_file = Path(data_file).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {data_file}\n"
            "Simpan skrip di folder utama hong_MSG3D, atau ubah DATA_FILE."
        )

    # Hanya baca sampel yang dipilih, tanpa memuat seluruh dataset ke RAM.
    data = np.load(data_file, mmap_mode="r", allow_pickle=False)
    if (
        data.ndim != 5
        or data.shape[1] != 3
        or data.shape[2] < 1
        or data.shape[3] != 25
        or data.shape[4] not in (1, 2)
        or data.dtype.kind not in "fiu"
    ):
        raise ValueError(
            "Input harus berupa data joint NTU berukuran (N,3,T,25,P), "
            f"P=1 atau 2. Ditemukan: shape={data.shape}, dtype={data.dtype}."
        )
    if not 0 <= sample_index < data.shape[0]:
        raise ValueError(
            f"SAMPLE_INDEX harus antara 0 dan {data.shape[0] - 1}."
        )

    if label_file is None:
        label_file = data_file.with_name(
            data_file.name.replace("_data_joint.npy", "_label.pkl")
        )
    metadata = read_action_metadata(label_file, sample_index, data.shape[0])

    X = np.array(data[sample_index], copy=True)
    if not np.isfinite(X).all():
        raise ValueError("Sampel mengandung NaN atau Inf. Pilih sampel lain.")
    if not np.any(X):
        raise ValueError("Sampel seluruhnya nol. Pilih SAMPLE_INDEX lain.")

    savemat(
        str(output_file),
        {"X": X, "sample_index_python": np.int32(sample_index), **metadata},
        appendmat=False,
        do_compression=True,
    )
    print(f"Dataset : {data_file}")
    print(f"Ukuran dataset (N,C,T,V,P): {data.shape}")
    print(f"Sampel ke-{sample_index + 1}: X berukuran {X.shape}")
    print(f"Nama rekaman: {metadata['sample_name']}")
    print(f"Aksi: {metadata['action_name']} ({metadata['action_code']})")
    print(f"Label Python: {int(metadata['action_label'])}")
    print(f"Berhasil dibuat: {output_file}")
    print("Salin ntu60_sample30.mat ke folder viewer MATLAB di Windows.")
    print("Perintah MATLAB: visualize_ntu_skeleton('ntu60_sample30.mat')")
    return output_file


if __name__ == "__main__":
    export_sample(DATA_FILE, SAMPLE_INDEX, OUTPUT_FILE, LABEL_FILE)
