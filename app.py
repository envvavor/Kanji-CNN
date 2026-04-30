from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

daftar_huruf = {
    0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お', 
    5: 'か', 6: 'き', 7: 'く', 8: 'け', 9: 'こ', 
    10: 'さ', 11: 'し', 12: 'す', 13: '君', 14: '日', 15: '本'
}
jumlah_kelas = len(daftar_huruf)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(jumlah_kelas, activation='softmax')
])

try:
    model.load_weights('model_kanji.keras')
    print("Model AI berhasil dimuat!")
except Exception as e:
    print("Gagal memuat weights:", str(e))

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': True,
        'supported_characters': list(daftar_huruf.values()),
        'total_classes': jumlah_kelas
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    target_char = data['target_character']
    image_b64 = data['image_base64']

    if target_char not in daftar_huruf.values():
        return jsonify({
            'success': True,
            'is_supported': False
        })

    try:
        # Decode gambar
        img_data = base64.b64decode(image_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Preprocessing
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = img.reshape(1, 64, 64, 1)

        # Tebak
        prediksi = model.predict(img)[0] # Ambil array probabilitas dari 16 huruf
        
        # Ambil 3 index dengan nilai tertinggi (diurutkan dari yang terbesar)
        top_3_indices = np.argsort(prediksi)[-3:][::-1]
        
        top_3_results = []
        for i in top_3_indices:
            top_3_results.append({
                'char': daftar_huruf.get(i, "?"),
                'prob': round(float(prediksi[i]) * 100, 1)
            })

        # Index tertinggi tetap di urutan ke-0 untuk patokan utama
        huruf_tebakan_ai = top_3_results[0]['char']
        akurasi_utama = top_3_results[0]['prob']
        is_match = (huruf_tebakan_ai == target_char)

        # Kirim data Top 3 ke Laravel
        return jsonify({
            'success': True,
            'predicted_char': huruf_tebakan_ai,
            'confidence': akurasi_utama,
            'is_match': is_match,
            'top_3': top_3_results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


    # Tebak hanya 1 karakter
        # prediksi = model.predict(img)
        # index_tertinggi = np.argmax(prediksi[0])
        # akurasi = float(prediksi[0][index_tertinggi]) * 100
        # huruf_tebakan_ai = daftar_huruf.get(index_tertinggi, "?")

        # is_match = (huruf_tebakan_ai == target_char)

        # return jsonify({
        #     'success': True,
        #     'predicted_char': huruf_tebakan_ai,
        #     'confidence': round(akurasi, 2),
        #     'is_match': is_match
        # }) hehe
