

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
DATA_FILE = PROJECT_DIR / "data" / "xsub" / "val_data_joint.npy"
LABEL_FILE = PROJECT_DIR / "data" / "xsub" / "val_label.pkl"
SAMPLE_INDEX = 60
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
                "Simpan viewer di folder utama hong_MSG3D, atau ubah "
                "DATA_FILE dan LABEL_FILE."
            )
    data = np.load(data_file, mmap_mode="r", allow_pickle=False)
    if (
        data.ndim != 5 or data.shape[0] < 1 or data.shape[1] != 3
        or data.shape[2] < 1 or data.shape[3] != 25
        or data.shape[4] not in (1, 2) or data.dtype.kind not in "fiu"
    ):
        raise ValueError(
            f"Input harus (N,3,T,25,P), P=1 atau 2; ditemukan {data.shape}."
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


class SkeletonViewer:
    def __init__(self, data_file, label_file, sample_index=60, fps=30):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS harus lebih besar dari 0.")
        self.data, self.names, self.labels = load_dataset(data_file, label_file)
        self.sample_count = self.data.shape[0]
        self.frame_count = self.data.shape[2]
        self.person_count = self.data.shape[4]
        self.paused = False
        self.sample_index = None
        self.animation = None

        self.fig = plt.figure(figsize=(8.6, 8.0), facecolor="#fafbfc")
        self.ax = self.fig.add_axes([0.04, 0.26, 0.92, 0.57], projection="3d")
        self.ax.set_facecolor("#fafbfc")
        self.ax.set_box_aspect((1, 1, 1))
        # self.ax.view_init(elev=12, azim=35)
        self.ax.view_init(elev=0, azim=-90)
        self.ax.set_proj_type("ortho")
        self.ax.set_xlabel(AXIS_NAMES[DISPLAY_ORDER[0]], labelpad=7)
        self.ax.set_ylabel(AXIS_NAMES[DISPLAY_ORDER[1]], labelpad=7)
        self.ax.set_zlabel(AXIS_NAMES[DISPLAY_ORDER[2]], labelpad=7)
        self.ax.tick_params(labelsize=9)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=4))
        self.title = self.fig.text(
            0.5, 0.95, "", ha="center", va="center",
            fontsize=17, weight="bold", color="#18324d",
        )
        self.sample_text = self.fig.text(
            0.5, 0.905, "", ha="center", fontsize=10, color="#34465a",
        )
        self.name_text = self.fig.text(
            0.5, 0.875, "", ha="center", fontsize=9, color="#617184",
        )
        self.status = self.fig.text(0.08, 0.21, "", fontsize=10, color="#b42318")
        self.bones, self.joints = [], []
        for person in range(self.person_count):
            color = PERSON_COLORS[person]
            bone, = self.ax.plot([], [], [], color=color, linewidth=2)
            joint, = self.ax.plot(
                [], [], [], linestyle="None", marker="o", markersize=4.8,
                color=color, label=f"Orang {person + 1}",
            )
            self.bones.append(bone)
            self.joints.append(joint)

        self.index_box = TextBox(
            self.fig.add_axes([0.15, 0.14, 0.11, 0.045]),
            "Indeks: ", initial=str(sample_index),
        )
        self.previous_button = Button(
            self.fig.add_axes([0.30, 0.14, 0.20, 0.045]), "Sampel sebelumnya",
        )
        self.next_button = Button(
            self.fig.add_axes([0.52, 0.14, 0.20, 0.045]), "Sampel berikutnya",
        )
        self.pause_button = Button(
            self.fig.add_axes([0.75, 0.14, 0.13, 0.045]), "Jeda",
        )
        self.slider = Slider(
            self.fig.add_axes([0.16, 0.075, 0.70, 0.028]),
            "Frame", 1, max(self.frame_count, 2), valinit=1,
            valstep=1, valfmt="%d",
        )
        self.slider.set_active(self.frame_count > 1)
        self.fig.text(
            0.5, 0.03,
            "Putar tampilan: seret mouse | Jeda atau lanjut: tombol / spasi",
            ha="center", fontsize=9, color="#617184",
        )
        self.select_sample(sample_index)
        self.index_box.on_submit(self._on_index)
        self.previous_button.on_clicked(lambda event: self.change_sample(-1))
        self.next_button.on_clicked(lambda event: self.change_sample(1))
        self.pause_button.on_clicked(lambda event: self.set_paused(not self.paused))
        self.slider.on_changed(self._on_frame)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Simpan objek animasi agar tetap hidup selama jendela terbuka.
        self.animation = FuncAnimation(
            self.fig, self._tick, init_func=self._draw,
            interval=1000 / fps, blit=False, cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    @property
    def artists(self):
        return tuple(self.bones + self.joints)

    def _set_index_box(self, index):
        old_events = self.index_box.eventson
        self.index_box.eventson = False
        try:
            self.index_box.set_val(str(index))
        finally:
            self.index_box.eventson = old_events

    def select_sample(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("Indeks sampel harus berupa integer.")
        index = int(index)
        if not 0 <= index < self.sample_count:
            raise ValueError(f"Indeks harus antara 0 dan {self.sample_count - 1}.")
        label = self.labels[index]
        if (
            isinstance(label, (bool, np.bool_))
            or not isinstance(label, (int, np.integer))
            or not 0 <= int(label) < len(ACTION_NAMES)
        ):
            raise ValueError("Label NTU60 harus berupa integer 0 sampai 59.")
        label = int(label)
        raw_name = self.names[index]
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        match = re.search(r"A(\d{3})(?=\.|$)", name)
        if match and int(match.group(1)) != label + 1:
            raise ValueError("Kode aksi pada nama rekaman tidak cocok dengan label.")

        X = np.array(self.data[index], copy=True)
        # Hanya seluruh slot tubuh yang nol dianggap kosong.
        # Joint valid yang tepat di origin tetap ditampilkan.
        active = np.any(np.isfinite(X) & (X != 0), axis=(0, 2))
        live_frames = np.flatnonzero(np.any(active, axis=1))
        if not live_frames.size:
            raise ValueError("Sampel ini tidak memiliki koordinat skeleton yang aktif.")
        points = np.transpose(X, (1, 3, 2, 0))[active].reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)][:, DISPLAY_ORDER]
        if not points.size:
            raise ValueError("Tidak ada joint dengan tiga koordinat yang valid.")
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        half_span = max(float(np.max(hi - lo)) * 0.58, 0.05)

        # Ganti state hanya setelah data dan label berhasil diperiksa.
        self.X, self.active = X, active
        self.sample_index = index
        self.first_frame = int(live_frames[0])
        self.last_frame = int(live_frames[-1])
        self.current_frame = self.first_frame
        self.ax.set_xlim(center[0] - half_span, center[0] + half_span)
        self.ax.set_ylim(center[1] - half_span, center[1] + half_span)
        self.ax.set_zlim(center[2] - half_span, center[2] + half_span)
        self.title.set_text(f"Aksi: {ACTION_NAMES[label]} (A{label + 1:03d})")
        self.sample_text.set_text(
            f"Sampel {index + 1} / {self.sample_count} | Indeks Python: {index}"
        )
        self.name_text.set_text(name)
        self.status.set_text("")
        self._set_index_box(index)
        present = np.flatnonzero(np.any(active, axis=0))
        self.ax.legend(
            handles=[self.joints[p] for p in present],
            loc="upper right", frameon=False, fontsize=9,
        )
        self.fig.canvas.manager.set_window_title(
            f"Skeleton NTU60 | {ACTION_NAMES[label]} | indeks {index}"
        )
        self._draw()
        self.fig.canvas.draw_idle()
        print(f"Indeks {index}: {name} | Aksi: {ACTION_NAMES[label]} (A{label + 1:03d})")

    def _draw(self):
        frame = self.current_frame
        for person in range(self.person_count):
            xyz = self.X[:, frame, :, person].T.astype(float, copy=True)[:, DISPLAY_ORDER]
            if not self.active[frame, person]:
                xyz[:] = np.nan
            xyz[~np.all(np.isfinite(xyz), axis=1)] = np.nan
            segments = np.full((3 * len(EDGES), 3), np.nan)
            segments[0::3] = xyz[EDGES[:, 0]]
            segments[1::3] = xyz[EDGES[:, 1]]
            self.bones[person].set_data_3d(*segments.T)
            self.joints[person].set_data_3d(*xyz.T)
        self.ax.set_title(
            f"Frame {frame + 1} / {self.frame_count}", fontsize=10, pad=5,
        )
        old_events = self.slider.eventson
        self.slider.eventson = False
        try:
            self.slider.set_val(frame + 1)
        finally:
            self.slider.eventson = old_events
        return self.artists

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
            self._set_index_box(self.sample_index)
            self.fig.canvas.draw_idle()

    def _on_index(self, text):
        try:
            index = int(text.strip())
        except ValueError:
            self.status.set_text("Isi kotak Indeks dengan bilangan bulat, misalnya 60.")
            self._set_index_box(self.sample_index)
            self.fig.canvas.draw_idle()
            return
        self._try_sample(index)

    def _on_frame(self, value):
        self.set_paused(True)
        self.current_frame = int(np.clip(int(value) - 1, 0, self.frame_count - 1))
        self._draw()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in (" ", "space"):
            self.set_paused(not self.paused)

    def _on_close(self, event):
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()


def main():
    # Buka jendela interaktif tersendiri saat dijalankan di Windows/PyCharm.
    if sys.platform == "win32":
        plt.switch_backend("TkAgg")
    viewer = SkeletonViewer(DATA_FILE, LABEL_FILE, SAMPLE_INDEX, FPS)
    print("Viewer siap. Gunakan tombol atau kotak Indeks untuk memilih rekaman.")
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
