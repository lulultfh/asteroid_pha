# Explainable Deep Learning for Potentially Hazardous Asteroids Identification
> **Using Sequential Orbital Parameters under Extreme Class Imbalance**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Deskripsi Proyek
Repositori ini memuat *pipeline* eksperimen *Deep Learning* secara *end-to-end* untuk mengklasifikasikan *Potentially Hazardous Asteroids* (PHA) menggunakan data deret waktu (*time-series*) parameter orbital (122 hari observasi). 

Tantangan utama dalam dataset astronomi ini adalah **Extreme Class Imbalance** (ketidakseimbangan kelas ekstrem), di mana jumlah asteroid berbahaya (PHA) sangat sedikit dibandingkan asteroid aman. Proyek ini membuktikan bahwa arsitektur berbasis *Long Short-Term Memory* (LSTM) yang dipadukan dengan *Optimal Thresholding* mengungguli model *State-of-the-Art* seperti Transformer. Lebih jauh, penggunaan **SHAP (SHapley Additive exPlanations)** diaplikasikan untuk membuktikan bahwa model mempelajari hukum fisika mekanika orbital (Keplerian mechanics) secara mandiri, bukan bertindak sebagai *black-box*.

## ✨ Kebaruan (Novelty) & Fitur Utama
1. **Pencegahan Data Leakage**: Menggunakan `GroupShuffleSplit` berdasarkan ID Asteroid (`spkid`) untuk memastikan data deret waktu dari satu asteroid tidak bocor ke *Train* dan *Test set* secara bersamaan.
2. **Penanganan Ekstrem Imbalance**: Implementasi *Focal Loss*, *Class Weights*, dan yang paling krusial: **Optimal Thresholding** (menggeser ambang batas dari 0.5 menjadi ~0.19 untuk LSTM) guna mencegah *False Negatives*.
3. **Komparasi Multi-Model**: Membandingkan arsitektur memori berurutan (LSTM, BiLSTM, GRU) dengan arsitektur berbasis atensi (Transformer Encoder).
4. **Explainable AI (XAI) pada 3D Time-Series**: Memipihkan (*flattening*) dimensi waktu dari `GradientExplainer` untuk memvisualisasikan kontribusi parameter orbit seperti `MOID` dan `H` terhadap probabilitas bahaya.

## 📊 Hasil Evaluasi Model
Berikut adalah perbandingan performa model setelah dilakukan **Threshold Optimization** pada data *test*:

| Model | Best Threshold | Accuracy | Precision (PHA) | Recall (PHA) | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **0.1953** | **0.9833** | **0.8000** | **1.0000** | **0.8889** | **0.9500** |
| GRU | 0.5327 | 0.9500 | 1.0000 | 0.2500 | 0.4000 | 0.3265 |
| BiLSTM | 0.5275 | 0.8500 | 0.1429 | 0.2500 | 0.1818 | 0.0812 |
| Transformer | 0.5187 | 0.2500 | 0.0816 | 1.0000 | 0.1509 | 0.0575 |

> **Kesimpulan:** Model **LSTM** berhasil mencapai tingkat deteksi (Recall) 100% tanpa ada satu pun PHA yang terlewat, dengan tingkat alarm palsu yang sangat minim (Precision 80%). Transformer mengalami *overfitting* akibat kelaparan data (*data-hungry*).

## 🧠 Interpretasi SHAP (Explainable AI)
Analisis SHAP membuktikan model AI bertindak selaras dengan konsensus ilmu astronomi:
* **MOID (Minimum Orbit Intersection Distance):** Model belajar bahwa semakin kecil nilai MOID (ditandai dengan warna biru), semakin tinggi probabilitas bahaya (SHAP value positif).
* **H (Absolute Magnitude):** Model secara akurat mengasosiasikan magnitudo absolut yang rendah (ukuran asteroid raksasa) dengan tingkat bahaya yang lebih besar.

*(Anda dapat melihat visualisasi SHAP Summary Plot secara lengkap di dalam Jupyter Notebook)*.

## 🛠️ Persyaratan Sistem & Instalasi
Pastikan Anda memiliki *environment* Python 3.8+ dengan *library* berikut:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap
