import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import time
from threading import Thread, Lock
from queue import Queue

# URL của server Flask (cập nhật nếu ngrok thay đổi)
SERVER_URL = "https://9302-34-143-198-177.ngrok-free.app/predict"

# Hàng đợi để lưu trữ khung hình và kết quả
frame_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=5)
lock = Lock()

def capture_frames():
    """Luồng để đọc khung hình từ webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    # Giảm độ phân giải và FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc khung hình")
                break

            # Bỏ khung hình nếu hàng đợi đầy
            if frame_queue.qsize() < 5:
                frame_queue.put(frame)
            else:
                continue

            time.sleep(0.05)  # ~20 FPS tối đa

    finally:
        cap.release()

def process_frames():
    """Luồng để gửi khung hình đến server và nhận kết quả"""
    while True:
        try:
            start_total = time.time()  # Đo thời gian toàn bộ
            frame = frame_queue.get()
            if frame is None:
                break

            # Chuyển khung hình thành JPG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_io = BytesIO(buffer)

            # Gửi đến server
            start = time.time()
            files = {'file': ('image.jpg', img_io, 'image/jpeg')}
            response = requests.post(SERVER_URL, files=files, timeout=3)
            print(f"Thời gian xử lý server: {time.time() - start:.3f} giây")

            if response.status_code == 200:
                img_response = Image.open(BytesIO(response.content))
                img_np = np.array(img_response)
                # Chuyển RGB sang BGR cho OpenCV
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                if result_queue.qsize() < 5:
                    result_queue.put(img_np)
            else:
                print(f"Lỗi từ server: {response.json().get('error', 'Unknown error')}")

            print(f"Thời gian xử lý toàn bộ: {time.time() - start_total:.3f} giây")
            frame_queue.task_done()

        except requests.RequestException as e:
            print(f"Lỗi kết nối: {e}")
            frame_queue.task_done()

def display_frames():
    """Luồng để hiển thị kết quả"""
    while True:
        try:
            img_np = result_queue.get()
            if img_np is None:
                break

            # Kiểm tra dữ liệu ảnh
            if img_np.size == 0 or img_np.shape[0] == 0:
                print("Ảnh rỗng hoặc lỗi")
                result_queue.task_done()
                continue

            with lock:
                cv2.imshow('Processed Frame', img_np)

            result_queue.task_done()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                frame_queue.put(None)
                result_queue.put(None)
                break

        except Exception as e:
            print(f"Lỗi hiển thị: {e}")
            result_queue.task_done()

def main():
    # Khởi tạo các luồng
    capture_thread = Thread(target=capture_frames)
    process_thread1 = Thread(target=process_frames)
    process_thread2 = Thread(target=process_frames)  # Luồng thứ hai
    display_thread = Thread(target=display_frames)

    # Đặt các luồng là daemon
    capture_thread.daemon = True
    process_thread1.daemon = True
    process_thread2.daemon = True
    display_thread.daemon = True

    # Bắt đầu các luồng
    capture_thread.start()
    process_thread1.start()
    process_thread2.start()
    display_thread.start()

    try:
        capture_thread.join()
        process_thread1.join()
        process_thread2.join()
        display_thread.join()

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()