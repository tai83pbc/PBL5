from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import pyngrok

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Thiết lập ngrok
pyngrok.ngrok.set_auth_token("2x2wnEbO497Dmjsu0UvzaswrzAS_36LKSEfd8h7rSv9jMkpBL")
public_url = pyngrok.ngrok.connect(5000)
print('Ngrok public URL:', public_url)


# Tải mô hình YOLOv8
model = YOLO("/content/best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "Empty filename"}, 400

    # Đọc ảnh
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_np = np.array(img)

    # Dự đoán với YOLOv8
    results = model(img_np, imgsz=320)  # Giảm imgsz xuống 320

    # Vẽ bounding box
    annotated_img = results[0].plot()

    # Chuyển sang Stuart images
    annotated_img_pil = Image.fromarray(annotated_img)

    # Lưu ảnh vào bộ nhớ đệm dưới dạng JPG
    img_io = io.BytesIO()
    annotated_img_pil.save(img_io, format='JPEG', quality=70)  # Giảm chất lượng xuống 70
    img_io.seek(0)

    return send_file(img_io, mimetype='image/webp')


if __name__ == '__main__':
    app.run(port=5000)