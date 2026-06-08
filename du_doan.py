import argparse
import os
import pickle
import random
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# TỰ ĐỊNH NGHĨA KHUNG XƯƠNG HAND CONNECTIONS THEO CHUẨN ĐIỂM 0-20 CỦA MEDIAPIPE
# Điều này giúp loại bỏ hoàn toàn việc import mp.solutions.hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón nhẫn
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Ngón út + gốc bàn tay
]


def get_random_image_paths(dataset_dir, num_samples=5):
    search_path = os.path.join(dataset_dir, '**', '*.jpg')
    all_images = glob(search_path, recursive=True)
    if not all_images:
        search_path_png = os.path.join(dataset_dir, '**', '*.png')
        all_images = glob(search_path_png, recursive=True)

    if len(all_images) < num_samples:
        raise ValueError(f"Không tìm đủ {num_samples} ảnh trong {dataset_dir}.")
    return random.sample(all_images, num_samples)


def normalize_landmarks(X):
    Xn = np.zeros_like(X, dtype=np.float32)
    for i, row in enumerate(X):
        pts = row.reshape(-1, 2).astype(np.float32)
        wrist = pts[0]
        rel = pts - wrist
        dists = np.linalg.norm(rel, axis=1)
        scale = dists.max()
        if scale <= 1e-6:
            scale = 1.0
        pts_norm = rel / scale
        Xn[i] = pts_norm.flatten()
    return Xn


def process_image_with_tasks_api(image_path, detector):
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        return None, None

    # Tasks API yêu cầu đối tượng mp.Image
    cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_rgb)
    
    # Nhận diện ảnh tĩnh bằng Tasks API (.detect)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.hand_landmarks:
        return None, cv_rgb

    # Lấy 21 điểm của bàn tay đầu tiên
    first_hand_lms = detection_result.hand_landmarks[0]
    
    raw_coords = []
    for lm in first_hand_lms:
        raw_coords.extend([lm.x, lm.y])
        
    return np.array(raw_coords, dtype=np.float32), cv_rgb, first_hand_lms


def draw_landmarks_pure_opencv(rgb_img, hand_landmarks):
    """Vẽ bộ khung xương thủ công bằng OpenCV, hoàn toàn không đụng tới mp.solutions"""
    annotated_image = rgb_img.copy()
    h, w, _ = annotated_image.shape
    
    # Chuyển đổi tọa độ Normalized (0.0 -> 1.0) sang dạng tọa độ Pixel thực tế
    pts = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append((cx, cy))

    # 1. Vẽ các đường nối xương khớp từ mảng tự định nghĩa (Màu xanh lá)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(pts) and end_idx < len(pts):
            cv2.line(annotated_image, pts[start_idx], pts[end_idx], (0, 255, 0), 2, cv2.LINE_AA)

    # 2. Vẽ các vòng tròn khớp điểm (Màu đỏ)
    for pt in pts:
        cv2.circle(annotated_image, pt, 4, (255, 0, 0), -1, cv2.LINE_AA)
            
    return annotated_image


def main(args):
    # 1. Load các mô hình học máy và bộ chuẩn hóa từ thư mục models
    print("Loading models and preprocessors...")
    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.joblib'))
    with open(os.path.join(args.model_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    ann_clf = joblib.load(os.path.join(args.model_dir, 'ann_model.joblib'))
    svm_clf = joblib.load(os.path.join(args.model_dir, 'svm_model.joblib'))

    # 2. Khởi tạo MediaPipe Tasks API (Chế độ IMAGE tĩnh)
    print("Khởi tạo MediaPipe Tasks Hand Landmarker...")
    base_options = python.BaseOptions(model_asset_path=args.model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Lấy ngẫu nhiên 5 ảnh từ dataset
    print(f"Quét ảnh ngẫu nhiên từ thư mục: {args.dataset_dir}...")
    try:
        image_paths = get_random_image_paths(args.dataset_dir, num_samples=5)
    except ValueError as e:
        print(e)
        return

    fig, axes = plt.subplots(5, 2, figsize=(11, 22))
    fig.suptitle('Inference on Images (Pure Tasks API): ANN vs SVM', fontsize=14, fontweight='bold')

    row_idx = 0
    for img_path in image_paths:
        true_label = os.path.basename(os.path.dirname(img_path))

        # Trích xuất dữ liệu bằng Tasks API
        res = process_image_with_tasks_api(img_path, detector)
        raw_landmarks, rgb_img, hand_lms = res[0], res[1], res[2] if len(res) == 3 else (None, res[1], None)

        if raw_landmarks is None:
            print(f"⚠️ Không phát hiện bàn tay: {os.path.basename(img_path)}")
            blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(blank_img, "No Hand Detected", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            axes[row_idx, 0].imshow(blank_img)
            axes[row_idx, 0].axis('off')
            axes[row_idx, 1].axis('off')
            axes[row_idx, 1].text(0.1, 0.5, "Không tìm thấy landmark tay.", color='red', fontsize=11)
            row_idx += 1
            continue

        # --- Chuẩn hóa dữ liệu tương tự file train ---
        landmark_norm = normalize_landmarks(raw_landmarks.reshape(1, -1))
        landmark_scaled = scaler.transform(landmark_norm)

        # ANN Predict
        ann_pred_idx = ann_clf.predict(landmark_scaled)
        ann_pred_label = le.inverse_transform(ann_pred_idx)[0]
        ann_prob = ann_clf.predict_proba(landmark_scaled).max() if hasattr(ann_clf, "predict_proba") else 1.0

        # SVM Predict
        svm_pred_idx = svm_clf.predict(landmark_scaled)
        svm_pred_label = le.inverse_transform(svm_pred_idx)[0]
        svm_prob = svm_clf.predict_proba(landmark_scaled).max() if hasattr(svm_clf, "predict_proba") else 1.0

        # --- Vẽ đồ thị bằng hàm OpenCV tự định nghĩa ---
        annotated_img = draw_landmarks_pure_opencv(rgb_img, hand_lms)

        axes[row_idx, 0].imshow(annotated_img)
        axes[row_idx, 0].set_title(f"Ảnh: {os.path.basename(img_path)} (Gốc: '{true_label}')", fontweight='bold', fontsize=10)
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].axis('off')
        ann_color = 'green' if str(ann_pred_label) == str(true_label) else 'red'
        svm_color = 'green' if str(svm_pred_label) == str(true_label) else 'red'

        axes[row_idx, 1].text(0.1, 0.65, f"ANN Predict: '{ann_pred_label}'", fontsize=13, color=ann_color, fontweight='bold')
        axes[row_idx, 1].text(0.1, 0.52, f"      Confidence: {ann_prob:.2%}", fontsize=10, color='gray')
        
        axes[row_idx, 1].text(0.1, 0.28, f"SVM Predict: '{svm_pred_label}'", fontsize=13, color=svm_color, fontweight='bold')
        axes[row_idx, 1].text(0.1, 0.15, f"      Confidence: {svm_prob:.2%}", fontsize=10, color='gray')

        print(f"Processed: {os.path.basename(img_path)} -> ANN: {ann_pred_label} | SVM: {svm_pred_label}")
        row_idx += 1

    detector.close()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'tasks_api_pure_inference.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n✅ Đã lưu kết quả so sánh hoàn chỉnh tại: {out_path}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on images using pure MediaPipe Tasks API')
    parser.add_argument('--dataset-dir', type=str, default='Dataset/test', help='Đường dẫn đến Dataset/test')
    parser.add_argument('--model-asset-path', type=str, default='./lib/hand_landmarker.task', help='Đường dẫn file .task')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--out-dir', type=str, default='inference_results')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)