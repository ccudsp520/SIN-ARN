r"""Visualisasi mask aktif Model_teacher pada proyek MSAG3D asli.

Simpan file ini di folder utama MSAG3D, sejajar dengan:
    train_action_pre_atu_msa.py
    unsupervised_globalgcn_3layer_action_KD_atu.py
Jalankan dengan Python yang sudah memiliki numpy dan matplotlib.

Kiri: skeleton lengkap, joint pilihan ditandai merah.
Kanan: skeleton setelah masking, joint pilihan dan tulang yang terhubung
       dengannya disembunyikan agar tidak muncul garis palsu ke origin.

Viewer membaca definisi self.parts dan assignment x/x_mask yang AKTIF
di Model_teacher.forward pada MODEL_FILE menggunakan AST, tanpa mengimpor
modul model. Baris komentar tidak dipakai. Tidak ada daftar joint buatan.
Versi asli yang diperiksa mengaktifkan self.parts[1], joint 5,6,7,8,22,23.
Hanya pola fixed-part asli yang didukung. Bila kode lokal mengaktifkan
mask acak atau pola lain, viewer berhenti dengan pesan yang jelas.

Mask direkonstruksi dari assignment tersebut, bukan mask dari rekaman
training terdahulu. Model neural, prediksi imputasi, dan checkpoint tidak
dijalankan. Dataset asli tetap read-only. Tidak memerlukan PyTorch/MATLAB.

M=1 berarti teramati; M=0 berarti dihapus; X_incomplete=M[None,...]*X.
M berukuran (T,25,2). Koordinat asli yang hilang hanya muncul di panel kiri.
Data default sama dengan alur train asli: ../hong_MSG3D/data/xsub.
Ubah DATA_FILE/LABEL_FILE bersama bila ingin memakai split train.

Setelah mengubah mask di file model, klik Baca ulang mask.
Kotak Indeks menerima indeks sampel Python mulai dari 0.
Slider Frame menjeda animasi; tombol Jeda/Lanjut atau spasi mengatur playback.
Tampilan awal dari depan, dengan sumbu Y vertikal dan proyeksi ortografis.

Referensi:
https://github.com/shahroudy/NTURGB-D#action-classes
https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn758662(v=ieb.10)
Microsoft memakai indeks 0-based; nomor joint NTU di viewer ini adalah indeks+1.
https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.view_init.html
"""

import ast
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, Slider, TextBox


# Pengaturan yang boleh diubah.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "unsupervised_globalgcn_3layer_action_KD_atu.py"
DATA_DIR = PROJECT_DIR.parent / "hong_MSG3D" / "data" / "xsub"
DATA_FILE = DATA_DIR / "val_data_joint.npy"
LABEL_FILE = DATA_DIR / "val_label.pkl"
SAMPLE_INDEX = 9
FPS = 30

# Urutan tampilan X, Z, Y; array dataset tidak diubah.
DISPLAY_ORDER = (0, 2, 1)
AXIS_NAMES = ("X", "Y", "Z")
PERSON_COLORS = ("#2563b8", "#d77724")

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


def read_active_mask(model_file):
    """Baca mask fixed-part aktif tanpa import/exec kode model atau PyTorch."""
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


class MSAG3DMaskViewer:
    def __init__(self, data_file, label_file, model_file, sample_index=9, fps=30):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS harus lebih besar dari 0.")
        self.active_mask = read_active_mask(model_file)
        self.data, self.names, self.labels = load_dataset(data_file, label_file)
        self.sample_count = self.data.shape[0]
        self.frame_count = self.data.shape[2]
        self.person_count = self.data.shape[4]
        self.sample_index = None
        self.paused = False
        self.animation = None
        self.fig = plt.figure(figsize=(12.4, 8.4), facecolor="#fafbfc")
        self.axes = [
            self.fig.add_axes([0.045, 0.29, 0.44, 0.51], projection="3d"),
            self.fig.add_axes([0.515, 0.29, 0.44, 0.51], projection="3d"),
        ]
        self.bones = [[], []]
        self.joints = [[], []]
        self.highlight = []
        for panel, ax in enumerate(self.axes):
            ax.set_facecolor("#fafbfc")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=0, azim=-90)
            ax.set_proj_type("ortho")
            ax.set_xlabel("X", labelpad=6)
            ax.set_ylabel("")
            ax.set_zlabel("Y", labelpad=6)
            ax.tick_params(labelsize=8)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=4))
            # Sumbu kedalaman menumpuk dalam tampilan depan; sembunyikan tick-nya.
            ax.set_yticks([])
            for person in range(self.person_count):
                color = PERSON_COLORS[person]
                bone, = ax.plot([], [], [], color=color, linewidth=2)
                joint, = ax.plot(
                    [], [], [], linestyle="None", marker="o", markersize=4.5,
                    color=color, label=f"Orang {person + 1}",
                )
                self.bones[panel].append(bone)
                self.joints[panel].append(joint)
        for person in range(self.person_count):
            mark, = self.axes[0].plot(
                [], [], [], linestyle="None", marker="o", markersize=7,
                color="#cf303a", label="Joint yang dihapus oleh mask aktif",
            )
            self.highlight.append(mark)
        self.title = self.fig.text(
            0.5, 0.955, "", ha="center", va="center", fontsize=17,
            weight="bold", color="#18324d",
        )
        self.sample_text = self.fig.text(0.5, 0.915, "", ha="center", fontsize=10)
        self.name_text = self.fig.text(
            0.5, 0.885, "", ha="center", fontsize=9, color="#617184",
        )
        self.frame_text = self.fig.text(0.5, 0.852, "", ha="center", fontsize=10)
        self.mask_text = self.fig.text(
            0.5, 0.261, "", ha="center", fontsize=10, color="#34465a",
        )
        self.status = self.fig.text(0.07, 0.226, "", fontsize=9, color="#b42318")
        self.reload_button = Button(
            self.fig.add_axes([0.13, 0.178, 0.22, 0.040]), "Baca ulang mask",
        )
        self.fig.text(
            0.39, 0.190, "Mask dibaca dari Model_teacher.forward pada file model asli",
            fontsize=9, color="#617184",
        )
        self.index_box = TextBox(
            self.fig.add_axes([0.13, 0.120, 0.10, 0.040]),
            "Indeks: ", initial=str(sample_index),
        )
        self.previous_button = Button(
            self.fig.add_axes([0.28, 0.120, 0.20, 0.040]), "Sampel sebelumnya",
        )
        self.next_button = Button(
            self.fig.add_axes([0.50, 0.120, 0.20, 0.040]), "Sampel berikutnya",
        )
        self.pause_button = Button(
            self.fig.add_axes([0.74, 0.120, 0.13, 0.040]), "Jeda",
        )
        self.slider = Slider(
            self.fig.add_axes([0.15, 0.065, 0.70, 0.025]),
            "Frame", 1, max(self.frame_count, 2), valinit=1, valstep=1, valfmt="%d",
        )
        self.slider.set_active(self.frame_count > 1)
        self.fig.text(
            0.5, 0.024,
            "Kiri: referensi lengkap | Kanan: input SIN setelah mask aktif | Spasi: jeda/lanjut",
            ha="center", fontsize=9, color="#617184",
        )
        self.select_sample(sample_index)
        self.index_box.on_submit(self._on_index)
        self.reload_button.on_clicked(self.reload_mask)
        self.previous_button.on_clicked(lambda event: self.change_sample(-1))
        self.next_button.on_clicked(lambda event: self.change_sample(1))
        self.pause_button.on_clicked(lambda event: self.set_paused(not self.paused))
        self.slider.on_changed(self._on_frame)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.animation = FuncAnimation(
            self.fig, self._tick, init_func=self._draw,
            interval=1000 / fps, blit=False, cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    @property
    def joint_text(self):
        return ", ".join(str(joint) for joint in self.active_mask.joint_numbers)

    @property
    def artists(self):
        return tuple(
            self.bones[0] + self.bones[1] + self.joints[0] + self.joints[1]
            + self.highlight
        )

    @staticmethod
    def _set_box(box, text):
        previous = box.eventson
        box.eventson = False
        try:
            box.set_val(str(text))
        finally:
            box.eventson = previous

    def select_sample(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("Indeks sampel harus berupa integer.")
        index = int(index)
        if not 0 <= index < self.sample_count:
            raise ValueError(f"Indeks harus antara 0 dan {self.sample_count - 1}.")
        label = self.labels[index]
        if (isinstance(label, (bool, np.bool_))
                or not isinstance(label, (int, np.integer))
                or not 0 <= int(label) < len(ACTION_NAMES)):
            raise ValueError("Label NTU60 harus berupa integer 0 sampai 59.")
        label = int(label)
        raw_name = self.names[index]
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        match = re.search(r"A(\d{3})(?=\.|$)", name)
        if match and int(match.group(1)) != label + 1:
            raise ValueError("Kode aksi pada nama rekaman tidak cocok dengan label.")
        X = np.array(self.data[index], copy=True)
        # Aktivitas tubuh dihitung dari X lengkap, bukan dari hasil masking.
        active = np.any(np.isfinite(X) & (X != 0), axis=(0, 2))
        live_frames = np.flatnonzero(np.any(active, axis=1))
        if not live_frames.size:
            raise ValueError("Sampel ini tidak memiliki skeleton yang aktif.")
        points = np.transpose(X, (1, 3, 2, 0))[active].reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)][:, DISPLAY_ORDER]
        if not points.size:
            raise ValueError("Tidak ada joint dengan tiga koordinat yang valid.")
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        half_span = max(float(np.max(hi - lo)) * 0.58, 0.05)
        M, X_incomplete = mask_sample(X, self.active_mask)
        self.X, self.M, self.X_incomplete, self.active = X, M, X_incomplete, active
        self.sample_index = index
        self.first_frame, self.last_frame = int(live_frames[0]), int(live_frames[-1])
        self.current_frame = self.first_frame
        for ax in self.axes:
            ax.set_xlim(center[0] - half_span, center[0] + half_span)
            ax.set_ylim(center[1] - half_span, center[1] + half_span)
            ax.set_zlim(center[2] - half_span, center[2] + half_span)
        self.title.set_text(f"Aksi: {ACTION_NAMES[label]} (A{label + 1:03d})")
        self.sample_text.set_text(
            f"Sampel {index + 1} / {self.sample_count} | Indeks Python: {index}"
        )
        self.name_text.set_text(name)
        self.status.set_text("")
        self._set_box(self.index_box, index)
        self.fig.canvas.manager.set_window_title(
            f"Mask aktif MSAG3D | {ACTION_NAMES[label]} | indeks {index}"
        )
        self._draw()
        self.fig.canvas.draw_idle()
        print(f"Indeks {index}: {name} | Aksi: {ACTION_NAMES[label]} (A{label + 1:03d})")
        print(f"Mask aktif: joint {self.joint_text or 'tidak ada'} | semua frame dan orang")

    def _draw(self):
        frame = self.current_frame
        for person in range(self.person_count):
            original = self.X[:, frame, :, person].T.astype(float, copy=True)[:, DISPLAY_ORDER]
            incomplete = self.X_incomplete[:, frame, :, person].T.astype(float, copy=True)[:, DISPLAY_ORDER]
            invalid = ~np.all(np.isfinite(original), axis=1)
            if not self.active[frame, person]:
                original[:] = np.nan
                incomplete[:] = np.nan
            original[invalid] = np.nan
            incomplete[invalid | ~self.M[frame, :, person]] = np.nan
            for panel, xyz in enumerate((original, incomplete)):
                segments = np.full((3 * len(EDGES), 3), np.nan)
                segments[0::3] = xyz[EDGES[:, 0]]
                segments[1::3] = xyz[EDGES[:, 1]]
                self.bones[panel][person].set_data_3d(*segments.T)
                self.joints[panel][person].set_data_3d(*xyz.T)
            marked = original.copy()
            marked[self.M[frame, :, person]] = np.nan
            self.highlight[person].set_data_3d(*marked.T)
        self.axes[0].set_title("Skeleton lengkap | Merah: joint yang dihapus", fontsize=11, pad=10)
        self.axes[1].set_title("Input SIN | Setelah mask aktif", fontsize=11, pad=10)
        self.frame_text.set_text(f"Frame {frame + 1} / {self.frame_count}")
        count = len(self.active_mask.missing_indices)
        self.mask_text.set_text(
            f"Joint dihilangkan: [{self.joint_text or 'tidak ada'}] | {count}/25 per orang "
            f"({100 * count / 25:.0f}%) | Seluruh frame"
        )
        previous = self.slider.eventson
        self.slider.eventson = False
        try:
            self.slider.set_val(frame + 1)
        finally:
            self.slider.eventson = previous
        return self.artists

    def reload_mask(self, event=None):
        try:
            active_mask = read_active_mask(self.active_mask.source_file)
            M, X_incomplete = mask_sample(self.X, active_mask)
        except (ValueError, SyntaxError, OSError) as error:
            self.status.set_text(str(error).splitlines()[0])
            print(error)
            self.fig.canvas.draw_idle()
            return
        self.active_mask, self.M, self.X_incomplete = active_mask, M, X_incomplete
        self.status.set_text("")
        print(f"Mask dibaca ulang: joint {self.joint_text or 'tidak ada'}")
        self._draw()
        self.fig.canvas.draw_idle()

    def _tick(self, unused=None):
        if not self.paused:
            self.current_frame = (
                self.first_frame if self.current_frame >= self.last_frame
                else self.current_frame + 1
            )
        return self._draw()

    def set_paused(self, paused):
        self.paused = bool(paused)
        self.pause_button.label.set_text("Lanjut" if self.paused else "Jeda")
        if self.animation is not None and self.animation.event_source is not None:
            if self.paused:
                self.animation.pause()
            else:
                self.animation.resume()
        self.fig.canvas.draw_idle()

    def change_sample(self, delta):
        self._try_sample(self.sample_index + delta)

    def _try_sample(self, index):
        try:
            self.select_sample(index)
        except (ValueError, TypeError) as error:
            self.status.set_text(str(error))
            self._set_box(self.index_box, self.sample_index)
            self.fig.canvas.draw_idle()

    def _on_index(self, text):
        try:
            index = int(text.strip())
        except ValueError:
            self.status.set_text("Isi kotak Indeks dengan bilangan bulat, misalnya 9.")
            self._set_box(self.index_box, self.sample_index)
            self.fig.canvas.draw_idle()
            return
        self._try_sample(index)

    def _on_frame(self, value):
        self.set_paused(True)
        self.current_frame = int(np.clip(int(value) - 1, 0, self.frame_count - 1))
        self._draw()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in (" ", "space") and not self.index_box.capturekeystrokes:
            self.set_paused(not self.paused)

    def _on_close(self, event):
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()


def main():
    if sys.platform == "win32":
        plt.switch_backend("TkAgg")
    viewer = MSAG3DMaskViewer(DATA_FILE, LABEL_FILE, MODEL_FILE, SAMPLE_INDEX, FPS)
    source = viewer.active_mask
    print(f"Sumber: {source.source_file}")
    print(f"Model_teacher.forward | part {source.part_ids} | baris mask {source.source_lines}")
    print("Viewer siap. Merah di kiri menunjukkan joint yang dihapus di panel kanan.")
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
