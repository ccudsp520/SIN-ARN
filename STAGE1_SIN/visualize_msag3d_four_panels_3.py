r"""Viewer SIN MSAG3D: complete, incomplete, predicted, repaired.

Simpan di root MSAG3D, lalu Run dengan interpreter yang sudah memiliki PyTorch,
torchvision, numpy, matplotlib, serta dependensi proyek. Pilih tombol P1-P7
atau MR 0.2-0.6 untuk menghitung perbaikan pada sampel yang sama. Model dimuat
sekali; kondisi yang sudah dihitung disimpan dan dapat dibuka kembali.
Tombol Sampel sebelumnya/berikutnya berpindah rekaman dengan kondisi masking
yang sama. Indeks Python dimulai dari 0; nomor sampel dimulai dari 1.
Nama aksi dan nama rekaman mengikuti sampel. Pergantian sampel kembali ke
frame pertama yang valid. Tiga sampel terakhir disimpan sementara di memori.

Definisi joint P1-P7 dibaca dari Model_teacher lokal. MR memilih int(25*MR)
joint tanpa pengulangan pada setiap frame, sama untuk kedua slot orang,
sesuai pola random-joint per-frame di kode proyek. MASK_SEED membuat pilihan
acak dapat diulang. MR 0.3 dan 0.5 masing-masing menghapus 7 dan 12 joint.
Viewer mengganti mask input saat inferensi, tanpa mengedit kode riset.

Prediksi memakai full_body + Decoder yang diisi bobot SIN secara strict.
Panel Predicted menampilkan keluaran SIN mentah untuk seluruh joint.
Panel Repaired hanya mengganti joint hilang dengan prediksi tersebut.
Pada Original, input SIN adalah skeleton lengkap tanpa masking.
Koordinat joint teramati, dataset, kode model, dan checkpoint tidak diubah.
Tidak ada training atau pemilihan epoch. CHECKPOINT_FILE memakai SIN E73 yang
pernah digunakan pada NTU60 X-Sub, bukan hasil pemilihan epoch baru.

Jika dataset, model dan checkpoint tersedia, Run membaca SAMPLE_INDEX terbaru
dan membuka viewer interaktif. Hasil tiap kondisi disimpan otomatis ke
RESULT_FILE. --view membuka hasil ekspor tanpa PyTorch. Dalam mode tersebut,
tombol kondisi yang belum diekspor dinonaktifkan. File .npz versi lama juga
dapat dibuka. Untuk mengekspor seluruh kondisi dari Ubuntu terlebih dahulu:

    python visualize_msag3d_four_panels.py --export-all --export-only
    python visualize_msag3d_four_panels.py --view /path/ntu60_sin_four_panels.npz

Empat panel memakai frame dan batas sumbu yang sama. Merah: joint biasa,
termasuk seluruh joint pada panel Predicted. Hijau: joint yang dihapus pada
panel Complete. Ungu: joint yang diisi SIN pada panel Repaired.
Label aksi dibaca dari dataset. Ekspor Original versi lama belum memiliki
prediksi SIN; panel Predicted memberi keterangan jika prediksi belum tersedia.
"""

import ast
import argparse
import hashlib
import importlib.util
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pickle
import os
import re
import sys
import tempfile

import numpy as np


# Pengaturan yang boleh diubah.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "unsupervised_globalgcn_3layer_action_KD_atu.py"
DATA_DIR = PROJECT_DIR.parent / "hong_MSG3D" / "data" / "xsub"
DATA_FILE = DATA_DIR / "val_data_joint.npy"
LABEL_FILE = DATA_DIR / "val_label.pkl"
SAMPLE_INDEX = 9
FPS = 30
MASK_CONDITION = "P1"  # Original, P1..P7, MR0.2..MR0.6.
MASK_SEED = 0

# SIN E73 yang digunakan pada pengujian NTU60 X-Sub sebelumnya.
# Lokasi relatif ini berlaku ketika script disimpan di root MSAG3D.
CHECKPOINT_FILE = PROJECT_DIR / "weight_hong" / "msag3d_epoch73.pth"
RESULT_FILE = PROJECT_DIR / "ntu60_sin_four_panels.npz"
EXPORT_ONLY = False  # True: simpan hasil tanpa membuka jendela (untuk Ubuntu).

# Urutan tampilan X, Z, Y; array dataset tidak diubah.
DISPLAY_ORDER = (0, 2, 1)
AXIS_NAMES = ("X", "Y", "Z")
BASE_JOINT_COLOR = "#cf303a"  # Merah: joint biasa dan prediksi mentah.
MISSING_JOINT_COLOR = "#16834b"  # Hijau: joint yang dihapus.
IMPUTED_JOINT_COLOR = "#8752c1"  # Ungu: joint yang diisi SIN.
PERSON_COLORS = (BASE_JOINT_COLOR, BASE_JOINT_COLOR)
PREDICTION_COLORS = PERSON_COLORS
CONDITIONS = ("Original",) + tuple(f"P{i}" for i in range(1,8)) + tuple(
    f"MR0.{i}" for i in range(2,7)
)
PART_NAMES = {
    "P1":"Lengan kiri", "P2":"Lengan kanan", "P3":"Kedua tangan",
    "P4":"Kedua kaki", "P5":"Batang tubuh",
    "P6":"Lengan dan kaki kiri", "P7":"Lengan dan kaki kanan",
}

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

# Topologi 25 joint NTU, dikonversi dari indeks 1-based ke indeks Python.
EDGES = np.array([
    [1, 2], [2, 21], [3, 21], [4, 3], [5, 21], [6, 5], [7, 6], [8, 7],
    [9, 21], [10, 9], [11, 10], [12, 11], [13, 1], [14, 13], [15, 14],
    [16, 15], [17, 1], [18, 17], [19, 18], [20, 19], [22, 23], [23, 8],
    [24, 25], [25, 12],
], dtype=int) - 1


def load_dataset(data_file, label_file):
    """Buka dataset tanpa memuat semua sampel skeleton ke RAM."""
    data_file = Path(data_file).expanduser().resolve()
    label_file = Path(label_file).expanduser().resolve()
    for file_path in (data_file, label_file):
        if not file_path.is_file():
            raise FileNotFoundError(
                f"File tidak ditemukan: {file_path}\n"
                "Simpan viewer di folder utama MSAG3D, atau ubah "
                "DATA_FILE dan LABEL_FILE."
            )
    data = np.load(data_file, mmap_mode="r", allow_pickle=False)
    if (
        data.ndim != 5 or data.shape[0] < 1
        or data.shape[1:] != (3, 300, 25, 2) or data.dtype.kind not in "fiu"
    ):
        raise ValueError(
            "Mask kode asli memakai (N,3,300,25,2); "
            f"ukuran dataset ditemukan {data.shape}."
        )
    with label_file.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("File label harus berisi (sample_names, labels).")
    names, labels = payload
    if len(names) != data.shape[0] or len(labels) != data.shape[0]:
        raise ValueError(
            "Jumlah nama dan label tidak sama dengan jumlah sampel. "
            "Gunakan file data dan label dari split yang sama."
        )
    return data, names, labels


@dataclass(frozen=True)
class ActiveMask:
    source_file: Path
    missing_indices: tuple
    part_ids: tuple
    source_lines: tuple
    sample_shape: tuple

    @property
    def joint_numbers(self):
        return tuple(index + 1 for index in self.missing_indices)


def _same_expr(node, expression):
    if node is None:
        return False
    expected = ast.parse(expression, mode="eval").body
    def shape(value):
        return ast.dump(value).replace("ctx=Store()", "ctx=Load()")
    return shape(node) == shape(expected)


def _part_id(node):
    if (isinstance(node, ast.Subscript)
            and _same_expr(node.value, "self.parts")):
        key = node.slice
        if isinstance(key, ast.Constant) and type(key.value) is int:
            return key.value
    return None


def _part_indices(node):
    # Definisi asli menggunakan nomor 1-based: np.array([...]) - 1.
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
            and _same_expr(node.right, "1")):
        raise ValueError("Definisi part harus berupa np.array([...]) - 1.")
    array_call = node.left
    if not (isinstance(array_call, ast.Call)
            and _same_expr(array_call.func, "np.array")
            and len(array_call.args) == 1 and not array_call.keywords):
        raise ValueError("Definisi part harus menggunakan daftar nomor joint literal.")
    numbers = ast.literal_eval(array_call.args[0])
    if not isinstance(numbers, (list, tuple)) or any(
        type(number) is not int or not 1 <= number <= 25 for number in numbers
    ):
        raise ValueError("Definisi part mengandung nomor joint di luar 1..25.")
    return tuple(sorted(set(number - 1 for number in numbers)))


def _mask_target(node):
    if not (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("x", "x_mask")):
        return None
    dims = node.slice
    if not isinstance(dims, ast.Tuple) or len(dims.elts) != 5:
        return None
    for axis in (0, 1, 2, 4):
        dim = dims.elts[axis]
        if not (isinstance(dim, ast.Slice)
                and dim.lower is None and dim.upper is None and dim.step is None):
            return None
    part = _part_id(dims.elts[3])
    return (node.value.id, part) if part is not None else None


def read_active_mask(model_file, part_override=None):
    """Validasi alur asli, lalu baca part aktif atau pilihan P1-P7 dari sumber."""
    model_file = Path(model_file).expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_file}\n"
            "Simpan viewer di folder utama MSAG3D, sejajar dengan file model asli."
        )
    tree = ast.parse(model_file.read_text(encoding="utf-8-sig"), filename=str(model_file))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Model_teacher"]
    if len(classes) != 1:
        raise ValueError("Harus ada tepat satu class Model_teacher pada file model.")
    methods = {node.name: node for node in classes[0].body
               if isinstance(node, ast.FunctionDef)}
    if "__init__" not in methods or "forward" not in methods:
        raise ValueError("Model_teacher harus memiliki __init__ dan forward.")

    # Hanya assignment tingkat utama yang aktif, bukan definisi dalam komentar.
    definitions = {}
    for statement in methods["__init__"].body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        part = _part_id(statement.targets[0])
        if part is not None:
            definitions[part] = statement.value

    forward = methods["forward"]
    boundary = next((i for i, node in enumerate(forward.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "GCN_feature"
                             for target in node.targets)), None)
    if boundary is None:
        raise ValueError("Batas masking sebelum assignment GCN_feature tidak ditemukan.")

    def unsupported(statement):
        raise ValueError(
            f"Pola masking aktif berbeda pada {model_file.name}, baris {statement.lineno}.\n"
            "Viewer ini mendukung mask fixed-part pada Model_teacher asli. "
            "Mask acak/per-frame memerlukan pembacaan x_mask dari alur model tersebut."
        )

    # Pernyataan tambahan pada kode asli tidak memengaruhi mask fixed-part.
    # Tetap periksa bentuknya agar viewer tidak diam-diam mengabaikan alur baru.
    original_temporaries = {
        "ori_data": "x.clone()",
        "joint_occlusion": "np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)",
        "num_parts": "13",
        "part_indices": "np.random.choice(num_parts, size=N, replace=True)",
    }
    zeroed = {"x": [], "x_mask": []}
    mask_created = False
    lines = []
    for statement in forward.body[:boundary]:
        if (isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            unsupported(statement)
        target = statement.targets[0]
        if _same_expr(target, "(N, C, T, V, M)") and _same_expr(statement.value, "x.size()"):
            continue
        if isinstance(target, ast.Name) and target.id in original_temporaries:
            if not _same_expr(statement.value, original_temporaries[target.id]):
                unsupported(statement)
            continue
        if isinstance(target, ast.Name) and target.id == "x_mask":
            if mask_created or not _same_expr(statement.value, "np.ones((x.shape[0], 3, 300, 25, 2))"):
                unsupported(statement)
            mask_created = True
            continue
        assignment = _mask_target(target)
        if assignment is None or not mask_created or not _same_expr(statement.value, "0"):
            unsupported(statement)
        array_name, part = assignment
        zeroed[array_name].append(part)
        if array_name == "x_mask":
            lines.append(statement.lineno)

    # Conversion ke tensor di akhir forward asli tidak mengubah nilai mask.
    # Assignment lain ke mask setelah encoder tidak boleh diabaikan.
    for statement in forward.body[boundary:]:
        for node in ast.walk(statement):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "x_mask":
                        if not _same_expr(node.value, "torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(device)"):
                            unsupported(node)
                    elif (isinstance(target, ast.Subscript)
                          and isinstance(target.value, ast.Name)
                          and target.value.id == "x_mask"):
                        unsupported(node)

    if (not mask_created or not zeroed["x_mask"]
            or zeroed["x"] != zeroed["x_mask"]):
        raise ValueError("Assignment penghapusan x dan x_mask harus memakai part yang sama.")
    part_ids = tuple(sorted(set(zeroed["x_mask"])))
    if part_override is not None:
        if type(part_override) is not int or not 1 <= part_override <= 7:
            raise ValueError("Pilihan part harus P1 sampai P7.")
        part_ids = (part_override,)

    # Definisi part yang dipakai harus tetap literal dan tidak ditimpa
    # oleh kondisi/loop atau helper yang dipanggil oleh constructor.
    init = methods["__init__"]
    top_level = {id(node) for node in init.body}
    for statement in ast.walk(init):
        if isinstance(statement, ast.Assign) and id(statement) not in top_level:
            if any(_part_id(target) in part_ids for target in statement.targets):
                unsupported(statement)
    helper_names = {
        node.func.attr for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
        and node.func.attr in methods
    }
    for helper_name in helper_names:
        for statement in ast.walk(methods[helper_name]):
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if (isinstance(target, ast.Subscript)
                            and _same_expr(target.value, "self.parts")
                            and (_part_id(target) is None or _part_id(target) in part_ids)):
                        unsupported(statement)

    missing = set()
    for part in part_ids:
        if part not in definitions:
            raise ValueError(f"Definisi aktif self.parts[{part}] tidak ditemukan di __init__.")
        value = definitions[part]
        # Part 8 pada kode asli merupakan array kosong (tanpa occlusion).
        if _same_expr(value, "np.array([], dtype=np.int32)"):
            indices = ()
        else:
            indices = _part_indices(value)
        missing.update(indices)
    return ActiveMask(model_file, tuple(sorted(missing)), part_ids,
                      tuple(lines), (3, 300, 25, 2))


def mask_sample(X, active_mask):
    if X.shape != active_mask.sample_shape:
        raise ValueError(
            f"Mask asli memakai {active_mask.sample_shape}; sampel ditemukan {X.shape}."
        )
    # M: (T,V,P). Assignment asli menghapus seluruh C, T dan P.
    M = np.ones(X.shape[1:], dtype=bool)
    if active_mask.missing_indices:
        M[:, np.array(active_mask.missing_indices), :] = False
    X_incomplete = np.where(M[np.newaxis, ...], X, 0)
    return M, X_incomplete


def repaired_sample(complete, mask, prediction):
    """Isi joint hilang, pertahankan koordinat joint teramati secara persis."""
    if prediction.shape != complete.shape or mask.shape != complete.shape[1:]:
        raise ValueError("Ukuran complete, mask dan prediction tidak cocok.")
    if mask.dtype != np.bool_:
        raise ValueError("Mask harus boolean: True=teramati, False=hilang.")
    if not np.all(np.isfinite(complete)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Skeleton atau prediksi berisi NaN/Inf.")
    return np.where(mask[np.newaxis, ...], complete, prediction)


def sin_constructor_specs(model_file):
    """Ambil constructor encoder/decoder yang benar-benar dipakai pada teacher."""
    tree = ast.parse(Path(model_file).read_text(encoding="utf-8-sig"))
    teacher = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "Model_teacher")
    init = next(node for node in teacher.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    components = {}
    full_body_ids = None
    for node in init.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if _same_expr(target, "self.FULL_BODY"):
            if (isinstance(node.value, ast.Call)
                    and _same_expr(node.value.func, "np.asarray")
                    and len(node.value.args) == 1 and not node.value.keywords):
                full_body_ids = ast.literal_eval(node.value.args[0])
        for name in ("full_body", "Decoder"):
            if _same_expr(target, "self." + name):
                value = node.value
                if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                    raise ValueError("Constructor SIN pada kode lokal memiliki bentuk berbeda.")
                components[name] = (
                    value.func.id,
                    tuple(ast.literal_eval(arg) for arg in value.args),
                    {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords
                     if kw.arg is not None},
                )
                if any(kw.arg is None for kw in value.keywords):
                    raise ValueError("Constructor dengan **kwargs dinamis belum didukung.")
    if set(components) != {"full_body", "Decoder"} or full_body_ids != list(range(1,26)):
        raise ValueError("Viewer membutuhkan full_body dan Decoder dari teacher asli, FULL_BODY=1..25.")

    # Pastikan jalur regresi yang aktif sama dengan jalur asli yang disederhanakan
    # menjadi Decoder(full_body(incomplete)). Hindari melewatkan operasi baru.
    forward = next(node for node in teacher.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    feature_index = next(i for i,node in enumerate(forward.body)
                         if isinstance(node,ast.Assign)
                         and any(_same_expr(target,"GCN_feature") for target in node.targets))
    output_index = next(i for i,node in enumerate(forward.body)
                        if isinstance(node,ast.Assign)
                        and any(_same_expr(target,"output") for target in node.targets))
    stages = forward.body[feature_index:output_index+1]
    expected = (
        ("GCN_feature", "torch.empty(N,M,276,T,V).cuda().to(torch.float32)"),
        ("GCN_feature[:,:,:,:,self.FULL_BODY-1]", "self.full_body(x[:,:,:,self.FULL_BODY-1,:])"),
        ("output", "self.Decoder(GCN_feature)"),
    )
    if len(stages) != len(expected) or any(
        not isinstance(node,ast.Assign) or len(node.targets) != 1
        or not _same_expr(node.targets[0],target) or not _same_expr(node.value,value)
        for node,(target,value) in zip(stages,expected)
    ):
        raise ValueError("Jalur encoder/decoder aktif berbeda dari kode MSAG3D asli yang diperiksa.")
    return components


def checkpoint_components(payload, is_tensor):
    """Buka state_dict standar, lalu pisahkan bobot dua komponen SIN aktif."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint harus berupa state_dict atau dict yang berisi state_dict.")
    containers = [key for key in ("state_dict", "model_state_dict")
                  if key in payload and isinstance(payload[key], Mapping)]
    if len(containers) > 1:
        raise ValueError("Ada beberapa state_dict pada checkpoint; pilih checkpoint model yang jelas.")
    state = payload[containers[0]] if containers else payload
    if not state or any(not isinstance(key,str) or not is_tensor(value)
                        for key,value in state.items()):
        raise ValueError("State_dict harus berisi pasangan nama parameter dan tensor.")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]:value for key,value in state.items()}
    parts = {}
    for name in ("full_body", "Decoder"):
        prefix = name + "."
        parts[name] = {key[len(prefix):]:value for key,value in state.items()
                       if key.startswith(prefix)}
        if not parts[name]:
            raise ValueError(
                f"Checkpoint tidak memiliki parameter {prefix}* yang diperlukan SIN asli.\n"
                "Pilih checkpoint SIN yang sesuai dengan definisi model ini."
            )
    return parts


class SinPredictor:
    """Muat full_body dan Decoder sekali, lalu gunakan untuk semua kondisi."""
    def __init__(self, model_file, checkpoint_file, device_name="auto"):
        checkpoint_file = Path(checkpoint_file).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_file}")
        specs = sin_constructor_specs(model_file)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Inferensi memerlukan PyTorch dan dependensi proyek MSAG3D. "
                "Gunakan interpreter yang telah dipakai menjalankan viewer SIN."
            ) from error
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ) if device_name == "auto" else torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak tersedia pada interpreter ini.")

        model_file = Path(model_file).resolve()
        project = model_file.parent
        # Kedua proyek memiliki package 'model'; gunakan package MSAG3D.
        root = str(project)
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0,root)
        existing = sys.modules.get("model")
        if existing is not None:
            locations = list(getattr(existing,"__path__",[]))
            if not locations or not all(Path(p).resolve() == (project/"model").resolve() for p in locations):
                raise RuntimeError("Package model dari proyek lain sudah dimuat. Jalankan script ini sebagai proses baru.")
        spec = importlib.util.spec_from_file_location("_msag3d_visualization_source",model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("Tidak bisa memuat definisi model asli.")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"Dependensi proyek belum tersedia: {error.name}. "
                "Gunakan interpreter PyCharm yang dipakai menjalankan riset MSAG3D."
            ) from error
        modules = {name:getattr(module,constructor)(*args,**kwargs)
                   for name,(constructor,args,kwargs) in specs.items()}
        payload = torch.load(checkpoint_file,map_location="cpu",weights_only=True)
        state_parts = checkpoint_components(payload,torch.is_tensor)
        for name,component in modules.items():
            try:
                component.load_state_dict(state_parts[name],strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Bobot {name} tidak cocok dengan model asli pada {model_file.name}.\n"
                    "Periksa checkpoint dan definisi model saat training.\n" + str(error)
                ) from error
            component.to(device).eval()
        self.torch,self.device,self.modules = torch,device,modules
        self.checkpoint_file = checkpoint_file
        print(f"SIN dimuat | device: {device} | checkpoint: {checkpoint_file.name}")

    def predict(self, incomplete):
        torch = self.torch
        # Copy menjaga input viewer tidak berubah oleh operasi in-place model.
        tensor = torch.from_numpy(np.array(incomplete[np.newaxis],dtype=np.float32,copy=True)).to(self.device)
        with torch.no_grad():
            features = self.modules["full_body"](tensor)
            feature_shape = (1,incomplete.shape[3],276,incomplete.shape[1],incomplete.shape[2])
            if not isinstance(features,torch.Tensor) or tuple(features.shape) != feature_shape:
                raise ValueError(f"Feature SIN harus {feature_shape}, ditemukan {getattr(features,'shape',None)}.")
            prediction = self.modules["Decoder"](features.contiguous())
        expected_shape = (1,) + incomplete.shape
        if not isinstance(prediction,torch.Tensor) or tuple(prediction.shape) != expected_shape:
            raise ValueError(f"Output SIN harus {expected_shape}, ditemukan {getattr(prediction,'shape',None)}.")
        result = prediction.detach().cpu().numpy()[0].copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Output model mengandung NaN/Inf.")
        return result


def predict_with_checkpoint(incomplete, model_file, checkpoint_file, device_name="auto"):
    return SinPredictor(model_file,checkpoint_file,device_name).predict(incomplete)


def condition_mask(shape, condition, part_masks=None, seed=0):
    """M berukuran (T,V,P); pilihan acak per-frame selalu dapat diulang."""
    if shape != (3,300,25,2):
        raise ValueError("Viewer NTU60 memerlukan sampel (3,300,25,2).")
    if condition not in CONDITIONS:
        raise ValueError(f"Kondisi tidak dikenal: {condition}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed harus integer 0 sampai 2**32-1.")
    mask = np.ones(shape[1:],dtype=bool)
    if condition.startswith("P"):
        if part_masks is None or part_masks.shape != (7,25) or part_masks.dtype != np.bool_:
            raise ValueError("Definisi P1-P7 dari model lokal belum tersedia.")
        mask[:] = part_masks[int(condition[1])-1][None,:,None]
    elif condition.startswith("MR"):
        rng = np.random.RandomState(seed)
        count = int(shape[2]*float(condition[2:]))
        for frame in range(shape[1]):
            joints = rng.choice(shape[2],count,replace=False)
            mask[frame,joints,:] = False
    return mask


def result_condition(result):
    if "mask_condition" in result:
        value = result["mask_condition"]
        if value.shape != () or value.dtype.kind not in "US":
            raise ValueError("mask_condition harus text scalar.")
        return str(value.item())
    if result["mask"].all():
        return "Original"
    parts = result.get("mask_part_ids",np.array([],dtype=int))
    if parts.shape == (1,) and parts.dtype.kind in "iu" and 1 <= int(parts[0]) <= 7:
        return f"P{int(parts[0])}"
    return "Tersimpan"


class RepairSession:
    """Satu sampel, satu checkpoint, dan cache terpisah untuk setiap mask."""
    def __init__(self,base,model_file=None,checkpoint_file=None,device_name="auto",
                 result_file=None,predictor=None):
        self.base = base
        self.X = base["complete"]
        self.part_masks = base.get("fixed_part_masks")
        self.seed = int(base.get("mask_seed",np.array(0)).item())
        self.model_file,self.checkpoint_file = model_file,checkpoint_file
        self.device_name,self.result_file = device_name,result_file
        self.predictor = predictor
        self.data_file = None
        self.label_file = None
        self.cache = {}
        self.current_condition = None

    @classmethod
    def from_dataset(cls,data_file,label_file,model_file,checkpoint_file,sample_index,
                     result_file,device_name="auto",seed=0):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed harus integer 0 sampai 2**32-1.")
        model_file = Path(model_file).expanduser().resolve()
        checkpoint_file = Path(checkpoint_file).expanduser().resolve()
        data_file = Path(data_file).expanduser().resolve()
        label_file = Path(label_file).expanduser().resolve()
        result_file = Path(result_file).expanduser().resolve()
        protected = {Path(p).expanduser().resolve()
                     for p in (data_file,label_file,model_file,checkpoint_file)}
        if result_file in protected or result_file.suffix.lower() != ".npz":
            raise ValueError("RESULT_FILE harus .npz, berbeda dari dataset/model/checkpoint.")
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_file}")
        part_masks = np.ones((7,25),dtype=bool)
        for part in range(1,8):
            spec = read_active_mask(model_file,part_override=part)
            part_masks[part-1,list(spec.missing_indices)] = False
        data,names,labels = load_dataset(data_file,label_file)
        if type(sample_index) is not int or not 0 <= sample_index < len(data):
            raise ValueError(f"SAMPLE_INDEX harus 0 sampai {len(data)-1}.")
        label = labels[sample_index]
        if isinstance(label,(bool,np.bool_)) or not isinstance(label,(int,np.integer)) or not 0 <= int(label) < 60:
            raise ValueError("Label NTU60 harus integer 0..59.")
        label = int(label)
        raw_name = names[sample_index]
        name = raw_name.decode("utf-8") if isinstance(raw_name,bytes) else str(raw_name)
        match = re.search(r"A(\d{3})(?=\.|$)",name)
        if match and int(match.group(1)) != label+1:
            raise ValueError("Nama sampel dan label aksi tidak cocok.")
        complete = np.array(data[sample_index],dtype=np.float32,copy=True)
        if not np.isfinite(complete).all() or not np.any(complete != 0):
            raise ValueError("Sampel tidak memiliki skeleton valid.")
        base = {
            "schema_version":np.array(2,dtype=np.int64),"complete":complete,
            "label":np.array(label,dtype=np.int64),"sample_index":np.array(sample_index,dtype=np.int64),
            "sample_count":np.array(len(data),dtype=np.int64),"sample_name":np.array(name),
            "checkpoint":np.array(str(checkpoint_file)),"model_source":np.array(str(model_file)),
            "checkpoint_size_bytes":np.array(checkpoint_file.stat().st_size,dtype=np.int64),
            "checkpoint_mtime_ns":np.array(checkpoint_file.stat().st_mtime_ns,dtype=np.int64),
            "model_source_sha256":np.array(hashlib.sha256(model_file.read_bytes()).hexdigest()),
            "data_source":np.array(str(Path(data_file).resolve())),
            "label_source":np.array(str(label_file)),
            "fixed_part_masks":part_masks,"mask_seed":np.array(seed,dtype=np.int64),
        }
        session = cls(base,model_file,checkpoint_file,device_name,result_file)
        session.data_file,session.label_file = data_file,label_file
        if result_file.is_file():
            try:
                previous = cls.from_archive(result_file)
                keys = ("complete","label","sample_index","sample_count","sample_name",
                        "checkpoint","checkpoint_size_bytes","checkpoint_mtime_ns",
                        "model_source_sha256","fixed_part_masks","mask_seed")
                if all(key in previous.base and np.array_equal(base[key],previous.base[key]) for key in keys):
                    session.cache = previous.cache
                    session.current_condition = previous.current_condition
                    print(f"Menggunakan {len(session.cache)} kondisi yang telah dihitung untuk sampel ini.")
            except (ValueError,OSError,KeyError) as error:
                print(f"Hasil lama tidak dapat dipakai; inferensi akan dihitung ulang: {error}")
        return session

    @classmethod
    def from_archive(cls,result_file):
        loaded = load_result(result_file)
        excluded = {"incomplete","mask","prediction","prediction_available","repaired","mask_condition",
                    "mask_part_ids","stored_conditions"}
        base = {key:value for key,value in loaded.items()
                if key not in excluded and not key.startswith("case_")}
        session = cls(base)
        if "stored_conditions" in loaded:
            for i,condition in enumerate(loaded["stored_conditions"].tolist()):
                session.cache[condition] = session._assemble(
                    condition,loaded[f"case_{i}_mask"],loaded[f"case_{i}_prediction"],
                    bool(loaded.get(f"case_{i}_prediction_available",
                                    np.array(not loaded[f"case_{i}_mask"].all())).item())
                )
        else:
            session.cache[result_condition(loaded)] = loaded
        session.current_condition = result_condition(loaded)
        return session

    def can_select(self,condition):
        return condition in self.cache or condition == "Original" or (
            condition in CONDITIONS and self.model_file is not None
            and self.checkpoint_file is not None
        )

    def _assemble(self,condition,mask,prediction,prediction_available=True):
        incomplete = np.where(mask[None],self.X,0)
        repaired = repaired_sample(self.X,mask,prediction)
        return dict(self.base,incomplete=incomplete,mask=mask,prediction=prediction,
                    prediction_available=np.array(prediction_available,dtype=bool),
                    repaired=repaired,mask_condition=np.array(condition),
                    mask_part_ids=np.array([int(condition[1])] if condition.startswith("P") else [],dtype=np.int64))

    def get_result(self,condition):
        can_predict = self.predictor is not None or (
            self.model_file is not None and self.checkpoint_file is not None
        )
        if condition in self.cache and (
            bool(self.cache[condition].get("prediction_available",np.array(True)).item())
            or not can_predict
        ):
            self.current_condition = condition
            return self.cache[condition]
        if not self.can_select(condition):
            raise RuntimeError("Kondisi ini belum diekspor. Buka viewer dengan checkpoint SIN untuk menghitungnya.")
        mask = condition_mask(self.X.shape,condition,self.part_masks,self.seed)
        incomplete = np.where(mask[None],self.X,0)
        if can_predict:
            if self.predictor is None:
                self.predictor = SinPredictor(self.model_file,self.checkpoint_file,self.device_name)
            prediction = self.predictor.predict(incomplete)
            prediction_available = True
        elif mask.all():
            # Mode offline: Original tetap tersedia, prediksi tidak dipalsukan.
            prediction = np.zeros_like(self.X)
            prediction_available = False
        else:
            raise RuntimeError("Prediksi SIN memerlukan checkpoint atau hasil yang sudah diekspor.")
        result = self._assemble(condition,mask,prediction,prediction_available)
        self.cache[condition] = result
        try:
            self.save(condition)
        except Exception:
            self.cache.pop(condition,None)
            raise
        self.current_condition = condition
        print(f"Kondisi selesai: {condition} | hasil tersimpan: {self.result_file or 'memori'}")
        return result

    def save(self,selected):
        if self.result_file is None:
            return
        result = dict(self.cache[selected])
        result["schema_version"] = np.array(2,dtype=np.int64)
        conditions = [name for name in CONDITIONS if name in self.cache]
        result["stored_conditions"] = np.array(conditions)
        for i,name in enumerate(conditions):
            result[f"case_{i}_mask"] = self.cache[name]["mask"]
            result[f"case_{i}_prediction"] = self.cache[name]["prediction"]
            result[f"case_{i}_prediction_available"] = self.cache[name]["prediction_available"]
        self.result_file.parent.mkdir(parents=True,exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.result_file.parent,suffix=".npz",delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,**result)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.result_file)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def export_result(data_file,label_file,model_file,checkpoint_file,sample_index,result_file,
                  device_name="auto",condition="P1",seed=0,export_all=False):
    session = RepairSession.from_dataset(data_file,label_file,model_file,checkpoint_file,
                                        sample_index,result_file,device_name,seed)
    session.get_result(condition)
    if export_all:
        for name in CONDITIONS:
            session.get_result(name)
        session.save(condition)
    return session.result_file


def load_result(result_file):
    path = Path(result_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hasil ekspor belum ditemukan: {path}")
    with np.load(path,allow_pickle=False) as archive:
        result = {key:np.array(archive[key],copy=True) for key in archive.files}
    required = {"schema_version","complete","incomplete","prediction","repaired","mask",
                "label","sample_index","sample_count","sample_name","checkpoint"}
    if not required.issubset(result):
        raise ValueError("File .npz bukan hasil ekspor viewer SIN ini.")
    def integer(key):
        value = result[key]
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError(f"Metadata {key} harus integer scalar.")
        return int(value)
    version = integer("schema_version")
    if version not in (1,2):
        raise ValueError("Versi file ekspor belum didukung.")
    label,index,count = integer("label"),integer("sample_index"),integer("sample_count")
    if not (0 <= label < 60 and 0 <= index < count):
        raise ValueError("Metadata label atau indeks sampel tidak valid.")
    for key in ("complete","incomplete","prediction","repaired"):
        array = result[key]
        if array.shape != (3,300,25,2) or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"Array {key} harus float finite dengan ukuran (3,300,25,2).")
    mask = result["mask"]
    if mask.shape != (300,25,2) or mask.dtype != np.bool_:
        raise ValueError("Mask hasil ekspor harus boolean (300,25,2).")
    # Ekspor Original dari viewer lama sengaja melewati inferensi SIN.
    result.setdefault("prediction_available",np.array(not mask.all(),dtype=bool))
    available = result["prediction_available"]
    if available.shape != () or available.dtype != np.bool_:
        raise ValueError("prediction_available harus boolean scalar.")
    if not np.array_equal(result["incomplete"],np.where(mask[None],result["complete"],0)):
        raise ValueError("Incomplete tidak sesuai mask dan complete.")
    expected = repaired_sample(result["complete"],mask,result["prediction"])
    if not np.array_equal(result["repaired"],expected):
        raise ValueError("Repaired tidak sesuai prediksi atau mengubah joint teramati.")
    for key in ("sample_name","checkpoint"):
        if result[key].shape != () or result[key].dtype.kind not in "US":
            raise ValueError(f"Metadata {key} harus text scalar.")
    name = str(result["sample_name"].item())
    match = re.search(r"A(\d{3})(?=\.|$)",name)
    if match and int(match.group(1)) != label+1:
        raise ValueError("Nama sampel dan label ekspor tidak cocok.")
    if "fixed_part_masks" in result:
        parts = result["fixed_part_masks"]
        if parts.shape != (7,25) or parts.dtype != np.bool_:
            raise ValueError("Definisi mask P1-P7 pada ekspor tidak valid.")
    if version == 2:
        condition = result_condition(result)
        seed = integer("mask_seed")
        if not 0 <= seed < 2**32:
            raise ValueError("Seed ekspor di luar rentang yang didukung.")
        names = result.get("stored_conditions")
        if (names is None or names.ndim != 1 or names.dtype.kind not in "US"
                or not 1 <= names.size <= len(CONDITIONS)):
            raise ValueError("Daftar kondisi ekspor tidak valid.")
        names = names.tolist()
        if len(set(names)) != len(names) or condition not in names or any(name not in CONDITIONS for name in names):
            raise ValueError("Nama kondisi ekspor tidak valid.")
        for i,name in enumerate(names):
            stored_mask = result.get(f"case_{i}_mask")
            stored_prediction = result.get(f"case_{i}_prediction")
            expected_mask = condition_mask(result["complete"].shape,name,result.get("fixed_part_masks"),seed)
            if (stored_mask is None or stored_mask.dtype != np.bool_
                    or not np.array_equal(stored_mask,expected_mask)):
                raise ValueError(f"Mask kondisi {name} pada ekspor tidak sesuai definisinya.")
            if (stored_prediction is None or stored_prediction.shape != (3,300,25,2)
                    or stored_prediction.dtype.kind != "f" or not np.isfinite(stored_prediction).all()):
                raise ValueError(f"Prediksi kondisi {name} pada ekspor tidak valid.")
            stored_available = result.setdefault(f"case_{i}_prediction_available",
                                                 np.array(not stored_mask.all(),dtype=bool))
            if stored_available.shape != () or stored_available.dtype != np.bool_:
                raise ValueError(f"Status prediksi kondisi {name} tidak valid.")
            if name == condition and (
                not np.array_equal(stored_mask,mask)
                or not np.array_equal(stored_prediction,result["prediction"])
                or not np.array_equal(stored_available,available)
            ):
                raise ValueError("Kondisi aktif tidak sesuai hasil yang disimpan.")
    return result


class RepairViewer:
    def __init__(self,result_file,fps=30,session=None,initial_condition=None):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.ticker import MaxNLocator
        from matplotlib.widgets import Button,Slider
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS harus lebih besar dari 0.")
        self.session = session if session is not None else RepairSession.from_archive(result_file)
        initial_condition = initial_condition or self.session.current_condition or MASK_CONDITION
        self.result = self.session.get_result(initial_condition)
        self.condition = result_condition(self.result)
        self.X = self.result["complete"]
        self.M = self.result["mask"]
        self.arrays = (self.X,self.result["incomplete"],
                       self.result["prediction"],self.result["repaired"])
        self.frame_count,self.person_count = self.X.shape[1],self.X.shape[3]
        self.active = np.any(self.X != 0,axis=(0,2))
        live = np.flatnonzero(np.any(self.active,axis=1))
        if not live.size:
            raise ValueError("Tidak ada skeleton aktif dalam hasil ekspor.")
        self.first_frame,self.last_frame = int(live[0]),int(live[-1])
        self.current_frame = self.first_frame
        self.paused = False
        self.animation = None
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix="sin_viewer")
        self.future = None
        self.pending_kind = None
        self.sample_sessions = OrderedDict([(int(self.result["sample_index"]),self.session)])
        self.resume_after = False
        self.closed = False
        self.fig = plt.figure(figsize=(17.2,8.8),facecolor="#fafbfc")
        self.axes = [self.fig.add_axes([0.010+panel*0.248,0.355,0.236,0.445],projection="3d")
                     for panel in range(4)]
        self.bones,self.joints = [[] for _ in self.axes],[[] for _ in self.axes]
        self.highlight,self.filled_bones = [],[]
        for panel,ax in enumerate(self.axes):
            ax.set_facecolor("#fafbfc")
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=0,azim=-90)
            ax.set_proj_type("ortho")
            ax.set_xlabel("X",labelpad=5)
            ax.set_ylabel("")
            ax.set_zlabel("Y",labelpad=5)
            ax.tick_params(labelsize=8)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_yticks([])
            for person in range(self.person_count):
                color = PREDICTION_COLORS[person] if panel == 2 else PERSON_COLORS[person]
                bone, = ax.plot([],[],[],color=color,linewidth=2)
                joints, = ax.plot([],[],[],linestyle="None",marker="o",
                                  markersize=4,color=color)
                self.bones[panel].append(bone)
                self.joints[panel].append(joints)
        self.original_marks,self.repaired_marks = [],[]
        for person in range(self.person_count):
            marked, = self.axes[0].plot([],[],[],linestyle="None",marker="o",
                                        markersize=6.5,color=MISSING_JOINT_COLOR)
            filled, = self.axes[3].plot([],[],[],linestyle="None",marker="o",
                                        markersize=6.5,color=IMPUTED_JOINT_COLOR)
            filled_bone, = self.axes[3].plot([],[],[],color=IMPUTED_JOINT_COLOR,linewidth=2.2)
            self.original_marks.append(marked)
            self.repaired_marks.append(filled)
            self.filled_bones.append(filled_bone)
        label = int(self.result["label"])
        index = int(self.result["sample_index"])
        count = int(self.result["sample_count"])
        self.action_text = self.fig.text(0.5,0.963,f"Aksi: {ACTION_NAMES[label]} (A{label+1:03d})",
                      ha="center",fontsize=17,weight="bold",color="#18324d")
        self.index_text = self.fig.text(0.5,0.926,f"Indeks: {index} (0-{count-1}) | Sampel: {index+1}/{count}",
                      ha="center",fontsize=10)
        self.name_text = self.fig.text(0.5,0.895,str(self.result["sample_name"].item()),ha="center",
                      fontsize=9,color="#617184")
        self.sample_buttons = {
            "previous":Button(self.fig.add_axes([0.030,0.911,0.190,0.040]),"Sampel sebelumnya"),
            "next":Button(self.fig.add_axes([0.780,0.911,0.190,0.040]),"Sampel berikutnya"),
        }
        self.sample_buttons["previous"].on_clicked(lambda event:self._change_sample(-1))
        self.sample_buttons["next"].on_clicked(lambda event:self._change_sample(1))
        self.frame_text = self.fig.text(0.5,0.874,"",ha="center",fontsize=10)
        for ax,title in zip(self.axes,("Complete\nHijau: joint yang dihapus",
                                      "Incomplete\nInput SIN",
                                      "Predicted\nMerah: prediksi SIN seluruh joint",
                                      "Repaired\nUngu: joint diisi SIN")):
            ax.set_title(title,fontsize=10.5,pad=9)
        self.prediction_note = self.axes[2].text2D(
            0.5,0.48,"Prediksi SIN belum diekspor\nuntuk kondisi ini.",
            transform=self.axes[2].transAxes,ha="center",va="center",
            fontsize=10,color=BASE_JOINT_COLOR,bbox=dict(facecolor="#fafbfc",edgecolor="none",pad=5)
        )
        self.mask_text = self.fig.text(0.5,0.328,"",ha="center",fontsize=10,color="#34465a")
        self.condition_text = self.fig.text(0.5,0.298,"",ha="center",fontsize=11,color="#18324d")
        checkpoint = str(self.result["checkpoint"].item()).replace("\\","/").rsplit("/",1)[-1]
        self.fig.text(0.5,0.268,"Checkpoint: "+checkpoint,ha="center",fontsize=9,color="#617184")
        self.mask_buttons = {}
        self.fig.text(0.03,0.212,"Bagian:",fontsize=10,color="#34465a")
        for i,name in enumerate(CONDITIONS[:8]):
            button = Button(self.fig.add_axes([0.145+i*0.106,0.196,0.095,0.041]),name)
            button.on_clicked(lambda event,name=name:self._select_condition(name))
            self.mask_buttons[name] = button
        self.fig.text(0.03,0.156,"Acak:",fontsize=10,color="#34465a")
        for i,name in enumerate(CONDITIONS[8:]):
            button = Button(self.fig.add_axes([0.145+i*0.126,0.140,0.115,0.041]),name.replace("MR","MR "))
            button.on_clicked(lambda event,name=name:self._select_condition(name))
            self.mask_buttons[name] = button
        self.pause_button = Button(self.fig.add_axes([0.807,0.140,0.13,0.041]),"Jeda")
        self.pause_button.on_clicked(lambda event:self.set_paused(not self.paused))
        self.status_text = self.fig.text(0.5,0.108,"",ha="center",fontsize=9,color="#617184")
        self.slider = Slider(self.fig.add_axes([0.15,0.061,0.70,0.024]),"Frame",
                             1,self.frame_count,valinit=self.current_frame+1,valstep=1,valfmt="%d")
        self.slider.on_changed(self._on_frame)
        self.fig.text(0.5,0.022,"Merah: joint biasa | Hijau: joint yang dihapus | Ungu: joint diisi SIN | Spasi: jeda/lanjut",
                      ha="center",fontsize=9,color="#617184")
        self.poll_timer = self.fig.canvas.new_timer(interval=100)
        self.poll_timer.add_callback(self._poll_result)
        self._set_bounds()
        self._refresh_buttons()
        self._set_ready_status()
        self.fig.canvas.manager.set_window_title(f"NTU60 Complete / Incomplete / Predicted / Repaired | indeks {index}")
        self.fig.canvas.mpl_connect("key_press_event",self._on_key)
        self.fig.canvas.mpl_connect("close_event",self._on_close)
        self._draw()
        self.animation = FuncAnimation(self.fig,self._tick,init_func=self._draw,
                                       interval=1000/fps,blit=False,cache_frame_data=False)

    def _set_bounds(self):
        # Prediksi di luar rentang GT ikut dimasukkan agar tidak terpotong.
        points = []
        arrays = [self.X,self.result["repaired"]]
        if bool(self.result["prediction_available"].item()):
            arrays.append(self.result["prediction"])
        for array in arrays:
            points.append(np.transpose(array,(1,3,2,0))[self.active].reshape(-1,3)[:,DISPLAY_ORDER])
        points = np.concatenate(points,axis=0)
        lo,hi = points.min(axis=0),points.max(axis=0)
        center = (lo+hi)/2
        span = max(float(np.max(hi-lo))*0.58,0.05)
        for ax in self.axes:
            ax.set_xlim(center[0]-span,center[0]+span)
            ax.set_ylim(center[1]-span,center[1]+span)
            ax.set_zlim(center[2]-span,center[2]+span)

    def _refresh_buttons(self):
        for name,button in self.mask_buttons.items():
            available = self.session.can_select(name)
            selected = name == self.condition
            enabled = available and self.future is None
            button.set_active(enabled)
            button.color = "#2563b8" if selected else ("#ffffff" if available else "#e9edf1")
            button.hovercolor = "#1b4c8e" if selected else "#dceaf9"
            button.ax.set_facecolor(button.color)
            button.label.set_color("white" if selected else ("#18324d" if available else "#9aa6b2"))
        self.pause_button.set_active(self.future is None)
        index,count = int(self.result["sample_index"]),int(self.result["sample_count"])
        live = self.session.data_file is not None and self.session.label_file is not None
        for name,button in self.sample_buttons.items():
            available = live and (index > 0 if name == "previous" else index < count-1)
            enabled = available and self.future is None
            button.set_active(enabled)
            button.color = "#ffffff" if enabled else "#e9edf1"
            button.hovercolor = "#dceaf9" if enabled else "#e9edf1"
            button.ax.set_facecolor(button.color)
            button.label.set_color("#18324d" if enabled else "#9aa6b2")

    def _set_ready_status(self):
        live = self.session.model_file is not None
        text = "Pilih sampel dengan tombol di atas. P1-P7 atau MR menentukan masking; indeks dimulai dari 0."
        if not live:
            text = "Mode hasil ekspor: pindah sampel memerlukan dataset dan checkpoint lokal. Tombol abu-abu belum tersedia."
        self.status_text.set_text(text)
        self.status_text.set_color("#617184")

    def _update_sample_info(self):
        label = int(self.result["label"])
        index,count = int(self.result["sample_index"]),int(self.result["sample_count"])
        self.action_text.set_text(f"Aksi: {ACTION_NAMES[label]} (A{label+1:03d})")
        self.index_text.set_text(f"Indeks: {index} (0-{count-1}) | Sampel: {index+1}/{count}")
        self.name_text.set_text(str(self.result["sample_name"].item()))
        self.fig.canvas.manager.set_window_title(
            f"NTU60 Complete / Incomplete / Predicted / Repaired | indeks {index}"
        )

    def _load_sample(self,index,condition,current_session):
        """Worker satu thread: muat sampel dan gunakan predictor yang sudah ada."""
        candidate = self.sample_sessions.get(index)
        if candidate is None:
            candidate = RepairSession.from_dataset(
                current_session.data_file,current_session.label_file,
                current_session.model_file,current_session.checkpoint_file,index,
                current_session.result_file,current_session.device_name,current_session.seed
            )
        if current_session.predictor is not None:
            candidate.predictor = current_session.predictor
        cached = condition in candidate.cache
        result = candidate.get_result(condition)
        if cached:
            # Saat kembali ke sampel tersimpan, ekspor mengikuti sampel di layar.
            candidate.save(condition)
        self.sample_sessions[index] = candidate
        self.sample_sessions.move_to_end(index)
        while len(self.sample_sessions) > 3:
            self.sample_sessions.popitem(last=False)
        return candidate,result

    def _change_sample(self,step):
        if (self.closed or self.future is not None or self.session.data_file is None
                or self.session.label_file is None):
            return
        index = int(self.result["sample_index"])+step
        if not 0 <= index < int(self.result["sample_count"]):
            return
        self.resume_after = not self.paused
        self.set_paused(True)
        self.status_text.set_text(f"Memuat indeks {index} dan menghitung SIN ({self.condition})... Tunggu hingga sampel tampil.")
        self.status_text.set_color("#a06416")
        self.pending_kind = "sample"
        self.future = self.executor.submit(self._load_sample,index,self.condition,self.session)
        self._refresh_buttons()
        self.poll_timer.start()
        self.fig.canvas.draw_idle()

    def _select_condition(self,condition):
        if self.closed or self.future is not None or not self.session.can_select(condition):
            return
        if condition == self.condition:
            return
        self.resume_after = not self.paused
        self.set_paused(True)
        if condition in self.session.cache:
            self._apply_result(self.session.get_result(condition))
            self.set_paused(not self.resume_after)
            return
        self.status_text.set_text(f"Menghitung prediksi SIN dan perbaikan {condition}... Tunggu hingga kondisi tampil.")
        self.status_text.set_color("#a06416")
        self.pending_kind = "condition"
        self.future = self.executor.submit(self.session.get_result,condition)
        self._refresh_buttons()
        self.poll_timer.start()
        self.fig.canvas.draw_idle()

    def _poll_result(self):
        if self.closed or self.future is None or not self.future.done():
            return True
        future,self.future = self.future,None
        kind,self.pending_kind = self.pending_kind,None
        self.poll_timer.stop()
        try:
            outcome = future.result()
            if kind == "sample":
                session,result = outcome
                self._apply_result(result,session=session,reset_frame=True)
            else:
                self._apply_result(outcome)
        except Exception as error:
            action = "Pergantian sampel" if kind == "sample" else "Perbaikan"
            print(f"{action} gagal: {error}",file=sys.stderr)
            message = (str(error).splitlines() or [type(error).__name__])[0]
            self.status_text.set_text(action+" gagal: "+message[:150]+" | Lihat terminal.")
            self.status_text.set_color("#bb2634")
            self._refresh_buttons()
        self.set_paused(not self.resume_after)
        return True

    def _apply_result(self,result,session=None,reset_frame=False):
        complete = result["complete"]
        active = np.any(complete != 0,axis=(0,2))
        live = np.flatnonzero(np.any(active,axis=1))
        if not live.size:
            raise ValueError("Tidak ada skeleton aktif dalam sampel.")
        if session is not None:
            self.session = session
        self.result = result
        self.X,self.active = complete,active
        self.frame_count,self.person_count = complete.shape[1],complete.shape[3]
        self.first_frame,self.last_frame = int(live[0]),int(live[-1])
        if reset_frame:
            self.current_frame = self.first_frame
        self.M = result["mask"]
        self.arrays = (self.X,result["incomplete"],result["prediction"],result["repaired"])
        self.condition = result_condition(result)
        self._update_sample_info()
        self._set_bounds()
        self._refresh_buttons()
        self._set_ready_status()
        self._draw()
        self.fig.canvas.draw_idle()

    @property
    def artists(self):
        return tuple(sum(self.bones,[])+sum(self.joints,[])+self.original_marks
                     +self.repaired_marks+self.filled_bones)

    @staticmethod
    def _segments(points,edges=EDGES):
        segments = np.full((3*len(edges),3),np.nan)
        segments[0::3] = points[edges[:,0]]
        segments[1::3] = points[edges[:,1]]
        return segments

    def _draw(self):
        frame = self.current_frame
        prediction_available = bool(self.result["prediction_available"].item())
        self.prediction_note.set_visible(not prediction_available)
        for person in range(self.person_count):
            observed = self.M[frame,:,person]
            coords = [array[:,frame,:,person].T.astype(float,copy=True)[:,DISPLAY_ORDER]
                      for array in self.arrays]
            if not self.active[frame,person]:
                for xyz in coords:
                    xyz[:] = np.nan
            coords[1][~observed] = np.nan
            if not prediction_available:
                coords[2][:] = np.nan
            for panel,xyz in enumerate(coords):
                edges = EDGES
                points = xyz.copy()
                if panel == 3:
                    edges = EDGES[np.all(observed[EDGES],axis=1)]
                    points[~observed] = np.nan
                self.bones[panel][person].set_data_3d(*self._segments(xyz,edges).T)
                self.joints[panel][person].set_data_3d(*points.T)
            original_mark = coords[0].copy()
            repaired_mark = coords[3].copy()
            original_mark[observed] = np.nan
            repaired_mark[observed] = np.nan
            self.original_marks[person].set_data_3d(*original_mark.T)
            self.repaired_marks[person].set_data_3d(*repaired_mark.T)
            filled_edges = EDGES[~np.all(observed[EDGES],axis=1)]
            self.filled_bones[person].set_data_3d(*self._segments(coords[3],filled_edges).T)
        self.frame_text.set_text(f"Frame {frame+1} / {self.frame_count}")
        missing = np.flatnonzero(~self.M[frame,:,0])+1
        text = ", ".join(str(int(joint)) for joint in missing) or "tidak ada"
        self.mask_text.set_text(f"Joint hilang/diisi: [{text}] | {len(missing)}/25 ({len(missing)*4}%)")
        description = PART_NAMES.get(self.condition,"Tanpa joint hilang" if self.condition == "Original" else "Mask hasil ekspor")
        if self.condition.startswith("MR"):
            description = f"Joint acak pada setiap frame | Seed: {self.session.seed}"
        self.condition_text.set_text(f"Kondisi: {self.condition} | {description}")
        old_events,old_draw = self.slider.eventson,self.slider.drawon
        self.slider.eventson,self.slider.drawon = False,False
        try:
            self.slider.set_val(frame+1)
        finally:
            self.slider.eventson,self.slider.drawon = old_events,old_draw
        return self.artists

    def _tick(self,unused=None):
        if not self.paused:
            self.current_frame = self.first_frame if self.current_frame >= self.last_frame else self.current_frame+1
        return self._draw()

    def set_paused(self,paused):
        self.paused = bool(paused)
        self.pause_button.label.set_text("Lanjut" if self.paused else "Jeda")
        if self.animation is not None and self.animation.event_source is not None:
            if self.paused:
                self.animation.pause()
            else:
                self.animation.resume()
        self.fig.canvas.draw_idle()

    def _on_frame(self,value):
        if self.future is not None:
            self.resume_after = False
        self.set_paused(True)
        self.current_frame = int(np.clip(int(value)-1,0,self.frame_count-1))
        self._draw()
        self.fig.canvas.draw_idle()

    def _on_key(self,event):
        if event.key in (" ","space") and self.future is None:
            self.set_paused(not self.paused)

    def _on_close(self,event):
        self.closed = True
        self.poll_timer.stop()
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=False,cancel_futures=True)
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Complete / Incomplete / Predicted / Repaired NTU60 memakai checkpoint SIN asli")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--view",type=Path,help="Buka .npz hasil ekspor tanpa memuat PyTorch")
    modes.add_argument("--checkpoint",type=Path,help="Checkpoint SIN yang dipilih pengguna")
    parser.add_argument("--sample-index",type=int,default=SAMPLE_INDEX)
    parser.add_argument("--data",type=Path,default=DATA_FILE)
    parser.add_argument("--labels",type=Path,default=LABEL_FILE)
    parser.add_argument("--model-file",type=Path,default=MODEL_FILE)
    parser.add_argument("--result",type=Path,default=RESULT_FILE)
    parser.add_argument("--device",default="auto",help="auto, cpu atau cuda:0")
    parser.add_argument("--mask",choices=CONDITIONS,help="Kondisi awal; default mengikuti MASK_CONDITION")
    parser.add_argument("--seed",type=int,default=MASK_SEED,help="Seed masking acak saat inferensi/ekspor")
    parser.add_argument("--export-only",action="store_true",help="Ekspor tanpa membuka jendela")
    parser.add_argument("--export-all",action="store_true",help="Hitung Original, P1-P7 dan MR0.2-0.6 untuk viewer offline")
    args = parser.parse_args(argv)
    export_only = args.export_only or EXPORT_ONLY
    checkpoint = args.checkpoint if args.checkpoint is not None else CHECKPOINT_FILE
    if args.view is not None:
        if args.export_only or args.export_all:
            parser.error("--view tidak dapat digabung dengan opsi ekspor.")
        result_file = args.view
        session = RepairSession.from_archive(result_file)
        export_only = False
    else:
        available = checkpoint is not None and all(
            Path(path).is_file() for path in (checkpoint,args.model_file,args.data,args.labels)
        )
        if args.checkpoint is not None or available or export_only or args.export_all:
            if checkpoint is None:
                parser.error("Isi CHECKPOINT_FILE atau gunakan --checkpoint PATH untuk inferensi SIN.")
            session = RepairSession.from_dataset(args.data,args.labels,args.model_file,checkpoint,
                                                args.sample_index,args.result,args.device,args.seed)
            condition = args.mask or MASK_CONDITION
            session.get_result(condition)
            if args.export_all:
                for i,name in enumerate(CONDITIONS,1):
                    print(f"Ekspor kondisi {i}/{len(CONDITIONS)}: {name}")
                    session.get_result(name)
                session.save(condition)
                session.current_condition = condition
            result_file = session.result_file
        elif args.result.is_file():
            result_file = args.result
            session = RepairSession.from_archive(result_file)
            arguments = sys.argv[1:] if argv is None else argv
            if any(value == "--sample-index" or value.startswith("--sample-index=") for value in arguments):
                parser.error("Mengganti sampel memerlukan dataset, model dan checkpoint lokal. Gunakan --view untuk membuka ekspor lama.")
            print(f"Membuka hasil ekspor: {result_file}")
        else:
            parser.error(
                "Dataset/model/checkpoint lokal belum tersedia. Simpan script di root MSAG3D "
                "dan isi CHECKPOINT_FILE, atau buka ekspor dengan --view PATH."
            )
    if args.mask is not None:
        session.get_result(args.mask)
    if export_only:
        return result_file
    import matplotlib.pyplot as plt
    if sys.platform == "win32":
        plt.switch_backend("TkAgg")
    viewer = RepairViewer(result_file,FPS,session=session)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
