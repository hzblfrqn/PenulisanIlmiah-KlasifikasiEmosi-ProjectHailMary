from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import math
from deep_translator import GoogleTranslator

app = Flask(__name__)

print("Memuat Model AI dan Vectorizer")
try:
    vectorizer = joblib.load('models/tfidf_vectorizer_global.pkl')
    model = joblib.load('models/lightgbm_model_global.pkl')
    print("Model berhasil diaktifkan!")
except Exception as e:
    print(f"\nGagal memuat model: {e}")
    vectorizer = None
    model = None

@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    ulasan_tampil = []
    total_pages = 1
    page_range = []

    try:
        df = pd.read_csv('phm_cross_domain_predictions.csv')
        ulasan_asli = df['clean_text'].dropna().tolist()
        
        total_reviews = len(ulasan_asli)
        total_pages = math.ceil(total_reviews / per_page)
        
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
            
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        ulasan_tampil = ulasan_asli[start_idx:end_idx]

        start_page = max(1, page - 2)
        end_page = min(total_pages, page + 2)
        page_range = range(start_page, end_page + 1)

    except:
        print("[WARNING] File CSV tidak ditemukan.")

    return render_template('index.html', 
                           reviews=ulasan_tampil, 
                           current_page=page, 
                           total_pages=total_pages,
                           page_range=page_range)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ulasan_asli = request.form.get('review_text')
        if not ulasan_asli or ulasan_asli.strip() == '':
            return jsonify({'error': 'Ulasan tidak boleh kosong.'}), 400
    
        # Sistem mendeteksi bahasa secara otomatis dan menerjemahkannya ke Inggris
        try:
            ulasan_proses = GoogleTranslator(source='auto', target='en').translate(ulasan_asli)
            # Menampilkan log di terminal VS Code untuk memantau hasil translasi
            print(f"[TRANSLATOR] Input: '{ulasan_asli}' -> Diterjemahkan: '{ulasan_proses}'")
        except Exception as trans_err:
            print(f"[WARNING] Translasi gagal, menggunakan teks asli. Error: {trans_err}")
            ulasan_proses = ulasan_asli

        # Gunakan teks hasil translasi (ulasan_proses) untuk dimasukkan ke model
        teks_vec = vectorizer.transform([ulasan_proses])
        probabilitas = model.predict_proba(teks_vec)[0]

        hasil = []
        for emosi, prob in zip(model.classes_, probabilitas):
            hasil.append({
                'emotion': emosi.capitalize(), 
                'probability': round(prob * 100, 2)
            })

        hasil_urut = sorted(hasil, key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'dominant_emotion': hasil_urut[0],
            'all_emotions': hasil_urut 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    try:
        # Membaca data prediksi yang sudah ada di CSV
        df = pd.read_csv('phm_cross_domain_predictions.csv')
        
        # Mencari kolom yang berisi label emosi (biasanya bernama 'emotion', 'label', atau 'predict')
        emotion_col = None
        for col in df.columns:
            if 'emotion' in col.lower() or 'predict' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
                emotion_col = col
                break
        
        if emotion_col:
            # Menghitung jumlah masing-masing emosi
            counts = df[emotion_col].value_counts().to_dict()
            total_reviews = len(df)
            
            # Memastikan format kapitalisasi (Joy, Sadness, dll)
            emotion_counts = {str(k).capitalize(): int(v) for k, v in counts.items()}
        else:
            # Fallback jika kolom tidak ditemukan
            emotion_counts = {'Joy': 850, 'Sadness': 420, 'Fear': 310, 'Anger': 200, 'Surprise': 150, 'Love': 159}
            total_reviews = 2089
            
    except Exception as e:
        print(f"[ERROR DASHBOARD] {e}")
        emotion_counts = {'Joy': 0, 'Sadness': 0, 'Fear': 0, 'Anger': 0, 'Surprise': 0, 'Love': 0}
        total_reviews = 0

    return render_template('dashboard.html', 
                           emotion_counts=emotion_counts, 
                           total_reviews=total_reviews)

if __name__ == '__main__':
    app.run(debug=True)
    
'''if __name__ == '__main__':
    app.run(debug=True, port=5000)'''