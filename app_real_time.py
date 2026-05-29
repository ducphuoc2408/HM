import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import pickle
import os
from collections import deque
from collections import Counter
import warnings
import time

# Bỏ qua cảnh báo phiên bản scikit-learn
warnings.filterwarnings("ignore")

# --- HÀM VẼ TÙY CHỈNH (THAY THẾ mp.solutions.drawing_utils) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),         # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),    # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20), # Ngón út
    (0, 17)                                 # Cạnh cườm tay
]

def draw_hand_custom(image, landmarks):
    h, w, _ = image.shape
    # Vẽ các đường nối
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
    # Vẽ các điểm mốc
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

# --- HÀM CHUẨN HÓA BẤT BIẾN (DÙNG CHUNG KHI TRAIN) ---
def extract_and_normalize(hand_landmarks, handedness_label):
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    coords = []
    for lm in hand_landmarks:
        coords.append([lm.x - wrist_x, lm.y - wrist_y])
    
    # Đối xứng gương trục X nếu là tay trái để đưa về định dạng tay phải lúc học
    if handedness_label == "Left":
        for i in range(21):
            coords[i][0] = -coords[i][0]
            
    flattened = [val for pt in coords for val in pt]
    max_val = max(abs(v) for v in flattened)
    if max_val == 0: max_val = 1.0
    
    normalized_vector = []
    for pt in coords:
        normalized_vector.extend([pt[0] / max_val, pt[1] / max_val])
    return normalized_vector


# 1. Tải mô hình ANN và Label Encoder
print("Đang tải mô hình máy học...")
try:
    ann_model = joblib.load('./models/ann_model.joblib')
    with open('./models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# Tải ảnh mẫu cho mỗi nhãn (nếu có)
sample_images = {}
dataset_train_dir = './Dataset/train'
for lbl in label_encoder.classes_:
    lbl_dir = os.path.join(dataset_train_dir, str(lbl))
    if os.path.isdir(lbl_dir):
        imgs = [f for f in os.listdir(lbl_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            img_path = os.path.join(lbl_dir, imgs[0])
            img = cv2.imread(img_path)
            if img is not None:
                sample_images[lbl] = cv2.resize(img, (140, 140))

# Temporal smoothing buffer giúp chống nhiễu, nhảy chữ liên tục khi quét webcam
WINDOW = 5
CONF_THRESHOLD = 0.6
pred_buffer = deque(maxlen=WINDOW)
conf_buffer = deque(maxlen=WINDOW)

# 2. Khởi tạo MediaPipe Tasks API (Hand Landmarker)
print("Đang tải MediaPipe Hand Landmarker...")
base_options = python.BaseOptions(model_asset_path='./lib/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Khởi động Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể kết nối với webcam.")
        break

    # Lật ảnh theo chiều ngang (như soi gương)
    frame = cv2.flip(frame, 1)
    
    # MediaPipe yêu cầu ảnh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Chuyển đổi khung hình thành đối tượng mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    
    # Xử lý nhận diện
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        # Lấy bàn tay đầu tiên xuất hiện trước webcam
        hand_landmarks = results.hand_landmarks[0]
        hand_label = results.handedness[0][0].category_name 
        
        # Vẽ bộ khung xương tùy chỉnh lên màn hình
        draw_hand_custom(frame, hand_landmarks)

        # Trích xuất đặc trưng bất biến chuẩn hóa (42 đặc trưng)
        features_list = extract_and_normalize(hand_landmarks, hand_label)
        X = np.array([features_list])

        try:
            # 5. Dự đoán bằng mô hình ANN
            pred_encoded = ann_model.predict(X)
            predicted_char = label_encoder.inverse_transform(pred_encoded)[0]
            
            # Tính toán độ tin cậy (Confidence)
            confidence = 0.0
            try:
                probs = ann_model.predict_proba(X)
                confidence = float(np.max(probs))
            except Exception:
                confidence = 1.0 # Fallback nếu model không hỗ trợ predict_proba

            # Thêm kết quả vào hàng đợi Smoothing
            pred_buffer.append(predicted_char)
            conf_buffer.append(confidence)

            # Khử nhiễu bằng thuật toán bầu chọn số đông (Majority Vote)
            most_common = Counter(pred_buffer).most_common(1)
            display_char = None
            display_conf = 0.0
            
            if most_common:
                cand, count = most_common[0]
                avg_conf = float(np.mean(conf_buffer)) if conf_buffer else 0.0
                # Chỉ hiển thị nếu độ tự tin trung bình vượt ngưỡng và có sự nhất quán trong buffer
                if avg_conf >= CONF_THRESHOLD and count >= max(2, int(0.6 * WINDOW)):
                    display_char = cand
                    display_conf = avg_conf

            if display_char is not None:
                conf_text = f'{display_conf*100:.1f}%'
                text = f'{display_char} ({conf_text})'
            else:
                text = 'Analyzing...'

            # Hiển thị kết quả text ra góc trái màn hình
            cv2.putText(frame, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Hiển thị ảnh mẫu (Sample Image) ở góc phải màn hình để đối chiếu học tập
            to_show = display_char if display_char is not None else predicted_char
            sample = sample_images.get(to_show)
            if sample is not None:
                sh, sw, _ = sample.shape
                fh, fw, _ = frame.shape
                x0 = fw - sw - 10
                y0 = 10
                frame[y0:y0+sh, x0:x0+sw] = sample
                cv2.rectangle(frame, (x0, y0), (x0+sw, y0+sh), (255, 255, 255), 2)
                
                conf_val_text = f'{confidence*100:.1f}%' if confidence is not None else ""
                label_text = f'Sample: {to_show} ({conf_val_text})'
                cv2.putText(frame, label_text, (fw - sw - 10, y0+sh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
        except Exception as e:
            cv2.putText(frame, "Loi logic du doan", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Giải phóng bộ đệm khi không có tay trong khung hình
        pred_buffer.clear()
        conf_buffer.clear()
        cv2.putText(frame, "Dang cho tay...", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị cửa sổ video
    cv2.imshow('Sign Language Recognition - MediaPipe Tasks', frame)

    # Bấm phím 'q' trên bàn phím để tắt webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên khi thoát
cap.release()
cv2.destroyAllWindows()