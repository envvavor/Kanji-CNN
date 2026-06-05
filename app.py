from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

daftar_huruf = {
    0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お',
    5: 'か', 6: 'き', 7: 'く', 8: 'け', 9: 'こ',
    10: 'さ', 11: 'し', 12: 'す', 13: 'せ', 14: 'そ',
    15: 'た', 16: 'ち', 17: 'つ', 18: 'て', 19: 'と',
    20: 'な', 21: 'に', 22: 'ぬ', 23: 'ね', 24: 'の',
    25: 'は', 26: 'ひ', 27: 'ふ', 28: 'へ', 29: 'ほ',
    30: 'ま', 31: 'み', 32: 'む', 33: 'め', 34: 'も',
    35: 'や', 36: 'ゆ', 37: 'よ', 38: 'ら', 39: 'り',
    40: 'る', 41: 'れ', 42: 'ろ', 43: 'わ', 44: 'を',
    45: 'ん', 46: 'ア', 47: 'イ', 48: 'ウ', 49: 'エ',
    50: 'オ', 51: 'カ', 52: 'キ', 53: 'ク', 54: 'ケ',
    55: 'コ', 56: 'サ', 57: 'シ', 58: 'ス', 59: 'セ',
    60: 'ソ', 61: 'タ', 62: 'チ', 63: 'ツ', 64: 'テ',
    65: 'ト', 66: 'ナ', 67: 'ニ', 68: 'ヌ', 69: 'ネ',
    70: 'ノ', 71: 'ハ', 72: 'ヒ', 73: 'フ', 74: 'ヘ',
    75: 'ホ', 76: 'マ', 77: 'ミ', 78: 'ム', 79: 'メ',
    80: 'モ', 81: 'ヤ', 82: 'ユ', 83: 'ヨ', 84: 'ラ',
    85: 'リ', 86: 'ル', 87: 'レ', 88: 'ロ', 89: 'ワ',
    90: 'ヲ', 91: 'ン', 92: '一', 93: '七', 94: '万',
    95: '三', 96: '上', 97: '下', 98: '中', 99: '九',
    100: '二', 101: '五', 102: '人', 103: '今', 104: '休',
    105: '何', 106: '先', 107: '入', 108: '八', 109: '六',
    110: '円', 111: '出', 112: '前', 113: '北', 114: '十',
    115: '千', 116: '午', 117: '半', 118: '南', 119: '友',
    120: '口', 121: '右', 122: '名', 123: '四', 124: '国',
    125: '土', 126: '外', 127: '大', 128: '天', 129: '女',
    130: '子', 131: '学', 132: '小', 133: '山', 134: '川',
    135: '左', 136: '年', 137: '後', 138: '日', 139: '時',
    140: '書', 141: '月', 142: '木', 143: '本', 144: '来',
    145: '東', 146: '校', 147: '母', 148: '毎', 149: '気',
    150: '水', 151: '火', 152: '父', 153: '生', 154: '男',
    155: '白', 156: '百', 157: '空', 158: '聞', 159: '花',
    160: '行', 161: '西', 162: '見', 163: '話', 164: '語',
    165: '読', 166: '車', 167: '金', 168: '長', 169: '間',
    170: '雨', 171: '電', 172: '食', 173: '高', 174: '魚'
}

jumlah_kelas = len(daftar_huruf)

mirip = {
    'へ': 'ヘ', 'ヘ': 'へ',
    '二': 'ニ', 'ニ': '二',
    '口': 'ロ', 'ロ': '口',
}

model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3,3), activation='relu'),
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
    model.load_weights('model_huruf_stage3kayanya.keras')
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
        prediksi = model.predict(img)[0] # Ambil array probabilitas
        
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

        pasangan_mirip = mirip.get(target_char)
        is_match = (huruf_tebakan_ai == target_char) or (huruf_tebakan_ai == pasangan_mirip)

        if is_match and huruf_tebakan_ai != target_char:
            huruf_tebakan_ai = target_char

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
