```markdown:README.md
# 🎬 Project Hail Mary - Emotion Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine%20Learning-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)

Repositori ini berisi implementasi sistem **Klasifikasi Emosi Multi-Kelas** berbasis *Natural Language Processing* (NLP) pada ulasan film *Project Hail Mary*. Proyek ini dikembangkan menggunakan **algoritma LightGBM** dan **pendekatan lintas domain (*Cross-Domain Modeling*)**, serta diintegrasikan ke dalam antarmuka aplikasi *web real-time* menggunakan **Flask**.

Proyek ini merupakan hasil penelitian skripsi/Penulisan Ilmiah di Universitas Gunadarma.

## 🚀 Fitur Utama
1. **Dynamic Web Scraping**: Mengekstrak data ulasan secara dinamis dari Letterboxd menggunakan `Selenium` dan `undetected-chromedriver` untuk menembus proteksi anti-bot *Cloudflare*.
2. **Cross-Domain Modeling**: Kecerdasan buatan dilatih menggunakan 16.000 data emosi publik dari *Hugging Face* (`dair-ai/emotion`) untuk mencegah bias subjektivitas, lalu diimplementasikan untuk memprediksi domain target (ulasan Letterboxd).
3. **Advanced NLP Pipeline**: Dilengkapi dengan pembersihan teks komprehensif, *Lemmatization*, dan modifikasi *Negation Handling* pada *stopword* untuk menjaga keutuhan polaritas emosi.
4. **Multilingual Proof of Concept (PoC)**: Sistem klasifikasi di *web* mampu mendeteksi bahasa input pengguna dan menerjemahkannya secara *real-time* ke bahasa Inggris menggunakan integrasi *deep-translator*.
5. **Interactive Analytics Dashboard**: Visualisasi sentimen audiens secara makro yang interaktif menggunakan `Chart.js`.

## 🛠️ Tech Stack
* **Bahasa Pemrograman:** Python, HTML, CSS, JavaScript
* **Web Framework:** Flask
* **Machine Learning:** LightGBM, Scikit-Learn (TF-IDF Vectorizer)
* **NLP & Text Processing:** NLTK, langdetect, deep-translator
* **Data Scraping:** Selenium, BeautifulSoup4
* **Data Visualisasi:** Chart.js, Seaborn, Matplotlib, WordCloud

## 📊 Kinerja Model
Pemodelan dieksekusi menggunakan representasi fitur numerik TF-IDF (10.000 *max features*, N-Gram 1-2). Algoritma LightGBM dikonfigurasi dengan *hyperparameter* `class_weight='balanced'` untuk menangani ketidakseimbangan kelas (*imbalanced data*).
* **Accuracy:** 87%
* **Precision (Weighted Avg):** 88%
* **Recall (Weighted Avg):** 87%
* **F1-Score (Weighted Avg):** 87%

## 📂 Struktur Repositori
```text
📦 PenulisanIlmiah-KlasifikasiEmosi-ProjectHailMary
 ┣ 📂 models/                 # File biner model AI yang telah dilatih
 ┃ ┣ 📜 lightgbm_model_global.pkl
 ┃ ┗ 📜 tfidf_vectorizer_global.pkl
 ┣ 📂 templates/              # File antarmuka HTML untuk Flask
 ┃ ┣ 📜 index.html            # UI Live Analyzer
 ┃ ┗ 📜 dashboard.html        # UI Analytics Dashboard
 ┣ 📜 app.py                  # Skrip utama Backend Flask
 ┣ 📜 phm_cross_domain_predictions.csv # Dataset hasil prediksi model
 ┣ 📜 README.md               # Dokumentasi proyek
 ┗ 📜 requirements.txt        # Daftar dependensi library

```

## ⚙️ Cara Instalasi & Menjalankan Aplikasi

1. **Kloning repositori ini**

```bash
   git clone [https://github.com/hzblfrqn/PenulisanIlmiah-KlasifikasiEmosi-ProjectHailMary.git](https://github.com/hzblfrqn/PenulisanIlmiah-KlasifikasiEmosi-ProjectHailMary.git)
   cd PenulisanIlmiah-KlasifikasiEmosi-ProjectHailMary
   

```

2. **Buat Virtual Environment (Opsional namun direkomendasikan)**

```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   

```

3. **Instal seluruh dependensi**

```bash
   pip install -r requirements.txt
   

```

4. **Jalankan aplikasi Flask**

```bash
   python app.py
   

```

5. **Buka di Web Browser**
Buka `http://localhost:5000` di *browser* Anda untuk mengakses antarmuka *Live Analyzer* dan *Dashboard*.

## 📝 Kesimpulan Penelitian

Hasil inferensi terhadap 2.089 ulasan Letterboxd menunjukkan bahwa audiens merespons narasi film *Project Hail Mary* dengan sentimen yang sangat positif, di mana spektrum emosi **Joy** mendominasi secara absolut, disusul oleh emosi *Sadness* dan *Anger* dengan margin frekuensi yang terpaut sangat jauh.

```
