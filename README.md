# Analisis Sentimen Maskapai AS dengan Deep Learning (NLP)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

Ini adalah *repository* untuk proyek *capstone* "Natural Language Processing (NLP) dengan Deep Learning" dari BISA AI Academy. Proyek ini bertujuan untuk membangun model *deep learning* yang mampu mengklasifikasikan sentimen (*tweet*) pelanggan terhadap maskapai penerbangan di Amerika Serikat ke dalam tiga kategori: positif, negatif, atau netral.

---

##  Daftar Isi

*   [Latar Belakang Proyek](#latar-belakang-proyek)
*   [Alur Kerja Proyek](#alur-kerja-proyek)
*   [Dataset](#dataset)
*   [Struktur Repository](#struktur-repository)
*   [Instalasi & Penggunaan](#instalasi--penggunaan)
*   [Hasil](#hasil)
*   [Pengembangan Selanjutnya](#pengembangan-selanjutnya)
*   [Lisensi](#lisensi)

---

## Latar Belakang Proyek

Di era digital, media sosial seperti Twitter menjadi platform utama bagi pelanggan untuk menyuarakan opini dan pengalaman mereka. Bagi industri penerbangan, umpan balik ini sangat berharga untuk meningkatkan kualitas layanan. Namun, volume data yang sangat besar membuatnya tidak mungkin untuk dianalisis secara manual.

Proyek ini mengatasi masalah tersebut dengan mengimplementasikan model *Long Short-Term Memory* (LSTM), sebuah arsitektur *Recurrent Neural Network* (RNN), untuk mengotomatisasi proses analisis sentimen *tweet* pelanggan.

---

## Alur Kerja Proyek

Proyek ini dibagi menjadi beberapa tahap utama yang terdokumentasi dalam *notebook*:

1.  **Eksplorasi Data (EDA)**: Memahami distribusi data, menganalisis isi *tweet*, dan visualisasi data sentimen per maskapai.
2.  **Pra-pemrosesan Teks**: Membersihkan data teks dengan menghilangkan URL, *mention*, tanda baca, angka, dan *stopwords* untuk menyiapkan data sebelum dimasukkan ke model.
3.  **Tokenisasi & Sequencing**: Mengubah teks menjadi urutan integer (*sequences*) yang dapat diproses oleh lapisan *Embedding* pada model.
4.  **Pemodelan**: Membangun arsitektur model *deep learning* menggunakan `TensorFlow/Keras` dengan lapisan `Embedding` dan `LSTM`.
5.  **Pelatihan Model**: Melatih model pada data training dan memvalidasinya untuk memastikan performa yang baik.
6.  **Evaluasi**: Mengukur performa model menggunakan metrik seperti *accuracy*, *precision*, *recall*, *F1-score*, dan menganalisis *confusion matrix*.

---

## Dataset

Dataset yang digunakan adalah **"Twitter US Airline Sentiment"** yang bersumber dari Kaggle. Dataset ini berisi ~14,640 *tweet* berbahasa Inggris yang telah diberi label sentimen.

*   **Fitur Kunci**: `text` (isi *tweet*)
*   **Target**: `airline_sentiment` (positif, negatif, netral)

---

## Struktur Repository

airline-sentiment-nlp-capstone/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


Untuk penjelasan detail mengenai setiap direktori, silakan lihat [di sini](#penjelasan-struktur). <!-- Atau Anda bisa menyalin penjelasan dari atas -->

---

## Instalasi & Penggunaan

Untuk mereplikasi proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/[NAMA_USER_ANDA]/airline-sentiment-nlp-capstone.git
    cd airline-sentiment-nlp-capstone
    ```

2.  **Buat dan aktifkan *virtual environment* (opsional, namun sangat disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal semua dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Dataset:**
    Unduh dataset dari [Kaggle: Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dan letakkan file `Tweets.csv` di dalam folder `data/raw/`.

5.  **Jalankan Jupyter Notebook:**
    Buka dan jalankan *notebook* di dalam folder `notebooks/` secara berurutan, dimulai dari `1.0_data_exploration_and_preprocessing.ipynb`.
    ```bash
    jupyter notebook
    ```

---

## Hasil

Model LSTM yang dikembangkan berhasil mencapai performa yang baik dalam mengklasifikasikan sentimen.

*   **Akurasi Pelatihan**: `[MASUKKAN AKURASI TRAINING DI SINI, misal: 95%]`
*   **Akurasi Validasi**: `[MASUKKAN AKURASI VALIDASI DI SINI, misal: 88%]`

Berikut adalah visualisasi *confusion matrix* dari performa model pada data uji:

![Confusion Matrix](reports/figures/confusion_matrix.png)
*(Catatan: Gambar ini akan muncul setelah Anda menghasilkan dan menyimpannya di folder yang benar)*

---

## Pengembangan Selanjutnya

Meskipun model saat ini sudah cukup baik, ada beberapa area untuk perbaikan di masa depan:
*   **Hyperparameter Tuning**: Menggunakan teknik seperti KerasTuner atau GridSearch untuk menemukan kombinasi *hyperparameter* terbaik.
*   **Menggunakan Model Transformer**: Mengimplementasikan arsitektur yang lebih canggih seperti BERT atau RoBERTa yang dapat memberikan pemahaman konteks yang lebih dalam.
*   **Penanganan *Class Imbalance***: Jika terdapat ketidakseimbangan kelas, teknik seperti *oversampling* (SMOTE) atau *class weights* dapat diterapkan.
*   **Deployment**: Menerapkan model sebagai REST API menggunakan Flask atau FastAPI agar dapat diintegrasikan dengan aplikasi lain.

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
