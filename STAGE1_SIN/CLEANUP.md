# Hasil pemilahan MSAG3D

Sumber pemeriksaan: ZIP MSAG3D yang dikirim setelah folder `weight_hong`
dipindahkan ke `MSAG3D_CHECKPOINTS`. ZIP berisi 395 file,
termasuk 156 file Python.

## Hasil

| Tujuan | File asli | Ukuran isi asli |
| --- | ---: | ---: |
| Repo GitHub | 50 | 0.616 MiB |
| Arsip lokal | 148 | 49.819 MiB |
| Dikeluarkan dari kedua paket | 197 | 20.824 MiB |

Repo memuat 43 file Python asli dan lima CSV historis.
Arsip menyimpan 113 file Python asli lainnya. Seluruh
156 file `.py` tetap tersimpan di salah satu paket.
Dokumen README, daftar dependensi, laporan ini, dan `.gitignore` ditambahkan
ke salinan repo. Isi seluruh kode Python asli dipertahankan byte per byte.

## Keputusan utama

| Isi | Keputusan | Alasan |
| --- | --- | --- |
| Trainer dan wrapper SIN MSA-G3D | Pertahankan | Alur asli `train_action_pre_atu_msa.py` |
| Model dan trainer Table I Row 1, 2, 3 | Pertahankan | Ablation MSA-AAGCN dengan kebijakan masking berbeda |
| Skrip evaluasi dan sweep yang tersedia | Pertahankan | Catatan reproduksi dan pengujian asli |
| `model/msg3d.py` | Pertahankan | Dipanggil oleh wrapper SIN asli untuk prediksi aksi |
| `model/msg3d2.py` | Pertahankan | Masih diimpor oleh `model/__init__.py` |
| Modul graph, loader, transform, dan encoder yang dirujuk | Pertahankan | Menjaga dependensi lokal |
| CSV `table1_*.csv` | Simpan di `results/` | Hasil historis, bukan cache |
| `visualize_msag3d_four_panels_3.py` | Pertahankan | Empat panel; warna merah, hijau, dan ungu |
| `visualize_msag3d_mask.py` | Pertahankan | Viewer mask ringan tanpa inferensi model |
| Empat viewer versi sebelumnya | Arsipkan | Fungsi utamanya tersedia pada viewer empat panel yang dipertahankan |
| `GLTA/` | Arsipkan sumber dan konfigurasi | Banyak salinan identik; ekstensi ShiftGCN tidak dirujuk alur terpilih |
| `Model22/` | Arsipkan sumber | Varian lama di luar import alur terpilih |
| Varian GCN, RA-GCN, dan utilitas lama yang tidak dirujuk | Arsipkan | Tetap tersedia untuk riset lanjutan |
| `xxxxxxxxxx_test` | Arsipkan | Log tensor prediksi, sekitar 40.6 MiB |
| `.npy` dan `.npz` | Arsipkan | Array eksperimen dan ekspor sampel visualisasi |
| Cache Python yang memiliki sumber | Keluarkan | Dapat dibuat ulang |
| Tiga cache tanpa pasangan sumber | Arsipkan | Menjaga kemungkinan pemulihan kode yang hilang |
| `.idea/` dan hasil build CUDA | Keluarkan | Metadata lokal dan hasil kompilasi lingkungan lama |

## Yang diperiksa

Seluruh 156 file Python diparsing untuk melihat definisi
dan import. Salinan kode identik dibandingkan menggunakan SHA-256.
Peninjauan fungsi difokuskan pada alur training, model, evaluasi, preprocessing,
dan visualisasi yang menentukan pilihan file. Ini bukan pembacaan manual
baris demi baris untuk semua file lama yang masuk arsip.

Dependensi lokal dari skrip yang dipertahankan ditelusuri sampai modul
pendukung, termasuk import relatif package, graph yang dipilih melalui string,
dan pemuatan model dinamis oleh viewer. Semua sumber Python di repo lolos
kompilasi sintaks. Seluruh file salinan dicocokkan dengan SHA-256 sumber ZIP.

Pemeriksaan ini tidak menjalankan training, evaluasi dataset, atau inferensi GPU.

## Hal yang masih perlu diperhatikan untuk menjalankan eksperimen

`Dataset/Datasets.py` membutuhkan `DLNest.Common.DatasetBase`, yang tidak ada
di ZIP ini. Beberapa skrip juga masih memakai path Linux asli dan `.cuda()`.
Lokasi checkpoint viewer dapat diarahkan melalui `--checkpoint` ke folder
`MSAG3D_CHECKPOINTS`; contoh tersedia di README.

Generator NTU yang disertakan mengeluarkan 150 frame, sementara MSA-G3D dan
viewer memakai 300 frame. Daftar epoch pada sweep lama juga belum mencakup
E61 sampai E80 satu per satu. Pengaturan dan definisi metrik asli dicatat
di README agar tidak tertukar dengan protokol eksperimen baru.

CSV Row 3 tersedia, tetapi skrip evaluasi pembuat CSV Row 3 tidak ada dalam
ZIP. Tidak ada kode pengganti yang dibuat selama pemilahan ini.

## Cara memakai hasil

Ekstrak paket repo di folder baru agar tidak menimpa proyek kerja.
Folder `MSAG3D` dari paket repo adalah salinan untuk GitHub.
Simpan `MSAG3D_ARCHIVE` dan `MSAG3D_CHECKPOINTS` sebagai folder terpisah di
luar repo. ZIP asli tetap menjadi cadangan lengkap, termasuk metadata yang
dikeluarkan dari dua paket hasil ini.

Arsip memuat `cleanup_manifest.json`: keputusan, alasan, ukuran, SHA-256,
dan lokasi tujuan untuk setiap file asli.

## Daftar sumber Python dalam repo

- `Dataset/Datasets.py`
- `Dataset/data_gen/gen_data_ntu120.py`
- `Dataset/data_gen/gen_data_ntu60.py`
- `Dataset/data_gen/preprocess.py`
- `GCN_Unsupervised/GCNEncoder/AGCNEncoder_3layer_action_atu.py`
- `GCN_Unsupervised/GCNEncoder/graph.py`
- `GCN_Unsupervised/rotation.py`
- `GCN_Unsupervised/unsupervised_globalgcn_3layer_action_KD_atu.py`
- `graph/__init__.py`
- `graph/kinetics.py`
- `graph/ntu_rgb_d.py`
- `graph/tools.py`
- `model/AGCNEncoder_3layer_action_atu.py`
- `model/__init__.py`
- `model/activation.py`
- `model/mlp.py`
- `model/ms_gcn.py`
- `model/ms_gtcn.py`
- `model/ms_tcn.py`
- `model/msag3d.py`
- `model/msg3d.py`
- `model/msg3d2.py`
- `model/va.py`
- `src/dataprocessor.py`
- `src/graph.py`
- `table1_row1_model.py`
- `table1_row2_model.py`
- `table1_row3_model.py`
- `test_pose2.py`
- `test_post_all.py`
- `test_table1_row1.py`
- `test_table1_row1_epoch_sweep.py`
- `test_table1_row1_extended_epoch_sweep.py`
- `test_table1_row2_epoch_sweep.py`
- `train_action_pre_atu_msa.py`
- `train_continue_table1_row1_to80.py`
- `train_table1_row1.py`
- `train_table1_row2.py`
- `train_table1_row3.py`
- `unsupervised_globalgcn_3layer_action_KD_atu.py`
- `utils.py`
- `visualize_msag3d_four_panels_3.py`
- `visualize_msag3d_mask.py`
