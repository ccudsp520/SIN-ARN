# Hasil pemilahan hong_MSG3D untuk SIN-ARN

Sumber: `hong_MSG3D_review_20261015_043505_541745e2.zip`, berisi 417 file.
Di antaranya ada 34 file Python proyek, 345 file dalam salinan Apex, dan
metadata paket pemeriksaan. ZIP ini tidak memuat dataset, checkpoint, atau
lima pickle skor teacher dari folder PC sebesar 21.36 GiB.

## Paket yang digunakan

`SIN-ARN_GitHub.zip` adalah paket untuk GitHub. Isinya folder `SIN-ARN`,
README dan `.gitignore` utama, `STAGE1_SIN` dari pemilahan sebelumnya,
serta `STAGE2_ARN` dari pemilahan ini. Stage 1 disalin tanpa perubahan isi.

`hong_MSG3D_ARCHIVE.zip` adalah arsip lokal. Simpan di luar repo.
Arsip ini memuat salinan Apex beserta lisensinya, template lama, modul yang
tidak digunakan, ekspor MAT, log, dan rekaman eksperimen kecil. Dataset dan
checkpoint besar tetap berada pada PC asal dan tidak ada dalam arsip ini.

## Keputusan

| Bagian | Keputusan | Alasan |
| --- | --- | --- |
| `main_TS.py`, model aktif, loader dan dependensi graph/convolution | Pertahankan | Jalur SIN-ke-ARN dengan KD dari skor teacher |
| `ensemble_TS.py` | Pertahankan | Fusion skor teacher historis, dengan formula asli |
| Viewer skeleton dan exporter MATLAB | Pertahankan | Alat bantu yang dipakai terpisah dari training |
| YAML TS train/test joint NTU60 CS | Pertahankan | Pengaturan launcher ARN asli |
| Snapshot model train, model test, launcher train dan dua config tersimpan | Pertahankan sebagai `experiment_snapshots/ntu60_cs/` | Masking dan pengaturan berbeda dari file utama |
| CSV per-epoch | Simpan di `results/` | Hasil historis asli |
| Snapshot launcher test | Arsipkan | Identik dengan `main_TS.py` di root |
| `model_bit.py`, `model_bit_pt.py` | Arsipkan | Pemeriksaan parameter lokal, bukan profiler lengkap eksperimen |
| `model/AAGCN_graph/dataprocessor.py`, `graph2.py` | Arsipkan | Tidak dirujuk jalur terpilih |
| `model/unsupervised_globalgcn_3layer_action_KD_atu.py` | Arsipkan | Modul lama dengan definisi `Model_student` ganda dan `MSA_G3D` tanpa import aktif |
| Template baseline, LF dan imputation lainnya | Arsipkan | Launcher sekarang mengimpor student dan loader secara langsung |
| Sepuluh YAML test baseline dengan data biner | Arsipkan sebagai byte asli | Tidak bisa dibaca sebagai YAML |
| `apex/` | Arsipkan sebagai dependensi eksternal | Salinan library NVIDIA, bukan implementasi SIN/ARN proyek |
| Dua `.mat`, log dan daftar prediksi sampel | Arsipkan | Dapat dirujuk kembali tanpa membebani source utama |

Ada 34 file asli yang disertakan di Stage 2 dan
389 file asli di arsip. Enam rekaman dalam `work_dir` juga
tersimpan sebagai snapshot/hasil di repo, agar arsip tetap memuat rekaman
`work_dir` kecil yang lengkap. Semua 417 file terpetakan, tanpa membuang
source dari ZIP pemeriksaan.

## Cakupan pembacaan dan pemeriksaan

Semua 34 file Python proyek diperiksa. File inti, loader, graph, utilitas,
viewer dan exporter dibaca melalui source. Salinan identik dicocokkan dengan
SHA-256. Versi hampir identik dibandingkan melalui diff lengkap atau definisi
AST untuk melihat perubahan yang aktif. Salinan library Apex diklasifikasikan
melalui import trainer dan metadata/library README; seluruh implementasi
internal Apex tidak dibaca baris demi baris.

Stage 2 memuat 28 file Python. Semua file salinan cocok byte per byte dengan
sumber, seluruh Python di repo gabungan lolos kompilasi sintaks, dan empat
YAML Stage 2 lolos parsing. Import lokal jalur aktif dan pola Git diperiksa.
Tidak ada refactor, perubahan masking, perubahan formula KD, training,
pemuatan checkpoint, evaluasi akurasi, atau inferensi CUDA dalam pemilahan ini.

## Batasan source asal

- Model utama hanya mengaktifkan P1. Snapshot train mencampur Original,
  P1-P7, dan MR0.2-MR0.6. Snapshot test menambahkan masking acak 60% di atas P1.
- Lima file skor teacher dimuat, tetapi T3 tidak dipakai pada fusion aktif.
  Kelima file tetap dibutuhkan trainer karena semuanya dibuka.
- Versi Windows mencatat Python 3.9.13 dan PyTorch 2.7.1 CPU. Ini bukan bukti
  environment CUDA yang dipakai training. Beberapa dependency trainer tidak
  tercatat dalam daftar metadata Windows tersebut.
- Model dan loader dipilih lewat import langsung. String model/feeder dalam
  YAML tidak menggantikan import tersebut. Wrapper saat ini memakai 60 kelas
  dan tensor dengan 300 frame.
- Skrip eksperimen baru CV, NTU120 XSUB dan XSET tidak ada di salinan ini.
  Template NTU120 yang tersedia adalah pengaturan baseline lama.
- Import `Model.va` dengan huruf besar ada pada metode MSAAGCN lama yang
  tidak dipakai jalur MSA-G3D utama. Masalah asal ini dicatat di README.
- Path dataset, skor teacher dan checkpoint masih memakai susunan folder
  asli. Baca README sebelum menjalankan eksperimen dari susunan SIN-ARN.

## Folder besar pada PC

| Isi di `D:\PROYEK6\IMPUTASI\hong_MSG3D` | Perlakuan |
| --- | --- |
| `data/` | Simpan sebagai dataset lokal di luar repo |
| `work_dir/` | Pertahankan checkpoint, optimizer states, skor dan log eksperimen di luar repo |
| `imputation_weight/` | Pertahankan checkpoint awal di luar repo |
| Lima `*_epoch1_test_score.pkl` di root | Pertahankan sebagai input KD di luar repo |
| `.venv/`, cache, metadata IDE, build | Tidak perlu diunggah ke GitHub |

Ekstrak `SIN-ARN_GitHub.zip` ke lokasi baru, atau salin isi folder hasilnya
ke folder SIN-ARN yang sudah disiapkan setelah membuat cadangan. Jangan
menaruh dataset, checkpoint, atau arsip lokal ke dalam paket upload source.
