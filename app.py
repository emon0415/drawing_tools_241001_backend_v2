from flask import Flask, request, jsonify
import base64
import os
from uuid import uuid4
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://drawing-tools-241001.vercel.app", "http://localhost:3000"]}})

#api_key = os.getenv('OPENAI_API_KEY')
#client = OpenAI(api_key=api_key)


#画像データを白黒の塗り絵に加工する関数
def image_process(img_binary):
    # バイナリデータをBytesIOに変換
    img_io = BytesIO(img_binary)
    
    # 画像ファイルをPIL形式で開く
    img_pil = Image.open(img_io)
    
    # PIL画像をOpenCV形式に変換
    img_array = np.array(img_pil)

    # グレースケール化(グレースケールで入力)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #ノイズ除去フィルタの追加
    img_blur =cv2.GaussianBlur(img_gray,(5,5),0)
    
    # アダプティブしきい値
    #img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)    
    
    # 膨張処理
    neiborhood = np.ones((5, 5), dtype=np.uint8)
    dilated = cv2.dilate(img_blur, neiborhood, iterations=1)

    # 差分取得
    diff = cv2.absdiff(dilated, img_gray)

    # エッジ検出（Cannyエッジ検出を使用）
    #edges= cv2.Canny(img_thresh, 100, 200)

    # 白黒反転
    result = 255 - diff

    # OpenCV形式の画像をPILに戻す
    result_img = Image.fromarray(result).convert('RGBA')

    return result_img


#画像をBase64にエンコードする
def pil_to_base64(img, format="PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return img_str

#画像サイズの調整とPNG形式への変換、バイナリデータとして保持
def prepare_image_for_dall_e(image):
    max_size = (1024, 1024)
    image.thumbnail(max_size, Image.LANCZOS)

    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        #print(request.json)  # 送信されたデータを確認
        data = request.json['image']
        # 先頭のdata:image/png;base64,などを除いた部分を取得
        file_data = data.split(',')[1]
        # 拡張子を取得
        file_extension = data.split('/')[1].split(';')[0]

        # uploadsディレクトリが存在しない場合、作成する
        if not os.path.exists('./uploads'):
            os.makedirs('./uploads')

        # base64デコード
        binary_data = base64.b64decode(file_data)
        
        # ファイルパス生成と保存
        file_name = f"{uuid4()}.{file_extension}"
        file_path = os.path.join('./uploads', file_name)
        
        with open(file_path, 'wb') as file:
            file.write(binary_data)
        
        # デバッグ用ログ
        print(f"アップロードされた画像の保存パス: {file_path}")
        
        #画像の加工関数実行
        img_process=image_process(binary_data)

        #加工後の画像を保存
        img_processed_path = os.path.join('./uploads', f"processed_{file_name}")
        img_process.save(img_processed_path, format="PNG")
        
        # デバッグ用ログ
        print(f"加工後の画像の保存パス: {img_processed_path}")
        
        #Base64へのエンコード
        img_base64 = pil_to_base64(img_process, format="PNG")

        #print(img_base64)
        return jsonify({
            "image":img_base64, 
            "message":"塗り絵加工しました！楽しんでね！",
            "processedImagePath":img_processed_path
        })

    except Exception as e:
        # エラーが発生した場合のデバッグログ
        print(f"アップロード処理でエラーが発生しました: {str(e)}")
        return jsonify({"message": f"エラーが発生しました: {str(e)}"}), 500


gunicorn app:app --bind 0.0.0.0:$PORT

#if __name__ == '__main__':
    #app.run(debug=True, host = '0.0.0.0')
