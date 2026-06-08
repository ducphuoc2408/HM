import os
import time
import pickle
import warnings
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore")

# --- CẤU HÌNH GIAO DIỆN WEB STREAMLIT ---
st.set_page_config(
    page_title="Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu Engine V3",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 AI Sign Language Recognition - High Performance")
st.markdown("Phiên bản tối ưu hóa hiệu năng: Kết quả dự đoán được **vẽ trực tiếp trên luồng Frame** giúp chống nghẽn và giật lag tuyệt đối khi chạy lại nhiều lần.")

# --- ĐỊNH NGHĨA KHUNG XƯƠNG HAND CONNECTIONS TỰ VẼ ---
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
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

# Hàm trích xuất đặc trưng hình học từ file gốc của bạn
def extract_and_normalize(hand_landmarks, handedness_label):
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    coords = []
    for lm in hand_landmarks:
        coords.append([lm.x - wrist_x, lm.y - wrist_y])
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

# --- CACHE LOAD MÔ HÌNH ---
@st.cache_resource
def load_ml_artifacts():
    ann = joblib.load('./models/ann_model.joblib')
    svm = joblib.load('./models/svm_model.joblib')
    scaler = joblib.load('./models/scaler.joblib')
    with open('./models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    base_options = python.BaseOptions(model_asset_path='./lib/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO
    )
    mp_detector = vision.HandLandmarker.create_from_options(options)
    return ann, svm, scaler, le, mp_detector

try:
    ann_model, svm_model, data_scaler, label_encoder, detector = load_ml_artifacts()
except Exception as e:
    st.error(f"Lỗi tải mô hình học máy: {e}")
    st.stop()

# Siêu tham số từ file chạy được của bạn
WINDOW = 5
CONF_THRESHOLD = 0.6

# Khởi tạo bộ đệm Deque nằm ở phạm vi đối tượng Transformer
ann_pred_buffer = deque(maxlen=WINDOW)
ann_conf_buffer = deque(maxlen=WINDOW)
svm_pred_buffer = deque(maxlen=WINDOW)
svm_conf_buffer = deque(maxlen=WINDOW)

# --- LỚP XỬ LÝ VIDEO WEBTRC ---
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Lật gương
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        ann_text = "ANN: Waiting..."
        svm_text = "SVM: Waiting..."
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            hand_label = results.handedness[0][0].category_name
            
            # 1. Vẽ khung xương
            draw_hand_custom(img, hand_landmarks)
            
            # 2. Trích xuất đặc trưng
            features_list = extract_and_normalize(hand_landmarks, hand_label)
            X_raw = np.array([features_list])
            X_scaled = data_scaler.transform(X_raw)
            
            try:
                # --- Dự đoán ANN ---
                ann_pred_enc = ann_model.predict(X_raw)
                ann_char = label_encoder.inverse_transform(ann_pred_enc)[0]
                ann_prob = float(np.max(ann_model.predict_proba(X_raw))) if hasattr(ann_model, "predict_proba") else 1.0
                
                ann_pred_buffer.append(ann_char)
                ann_conf_buffer.append(ann_prob)
                ann_vote = Counter(ann_pred_buffer).most_common(1)
                
                if ann_vote and np.mean(ann_conf_buffer) >= CONF_THRESHOLD:
                    ann_text = f"ANN: {ann_vote[0][0]} ({np.mean(ann_conf_buffer)*100:.1f}%)"
                else:
                    ann_text = "ANN: Analyzing..."

                # --- Dự đoán SVM ---
                svm_pred_enc = svm_model.predict(X_scaled)
                svm_char = label_encoder.inverse_transform(svm_pred_enc)[0]
                svm_prob = float(np.max(svm_model.predict_proba(X_scaled))) if hasattr(svm_model, "predict_proba") else 1.0
                
                svm_pred_buffer.append(svm_char)
                svm_conf_buffer.append(svm_prob)
                svm_vote = Counter(svm_pred_buffer).most_common(1)
                
                if svm_vote and np.mean(svm_conf_buffer) >= CONF_THRESHOLD:
                    svm_text = f"SVM: {svm_vote[0][0]} ({np.mean(svm_conf_buffer)*100:.1f}%)"
                else:
                    svm_text = "SVM: Analyzing..."

            except Exception:
                ann_text = "ANN: Error"
                svm_text = "SVM: Error"
        else:
            ann_pred_buffer.clear()
            ann_conf_buffer.clear()
            svm_pred_buffer.clear()
            svm_conf_buffer.clear()
            ann_text = "ANN: Waiting for hand..."
            svm_text = "SVM: Waiting for hand..."

        # VẼ KẾT QUẢ TRỰC TIẾP LÊN KHUNG HÌNH (ON-FRAME RENDERING)
        # Nền chữ đổ bóng màu đen
        cv2.putText(img, ann_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, svm_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        
        # Chữ chính màu sắc nổi bật (ANN màu xanh dương, SVM màu vàng cam)
        cv2.putText(img, ann_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(img, svm_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
        
        return img

# --- HIỂN THỊ CAMERA LÊN WEB ---
st.subheader("🎥 Live Comparison Stream")
ctx = webrtc_streamer(
    key="sign-language-pure-onframe",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Thêm hướng dẫn sử dụng nhỏ gọn, sạch sẽ ở dưới
st.info("💡 Mẹo: Kết quả hiển thị trực tiếp ở góc trên bên trái màn hình camera. Nếu bạn bấm Stop và Start lại, bộ nhớ đệm sẽ tự động làm sạch mà không gây lag.")