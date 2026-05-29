import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./lib/hand_landmarker.task"),
    num_hands=1, # Tập trung tối ưu xuất sắc cho 1 bàn tay kí hiệu của ASL-HG
    min_hand_detection_confidence=0.4,
    running_mode=VisionRunningMode.IMAGE
)


def extract_and_normalize(hand_landmarks, handedness_label):
    # 1. Dịch tâm về cổ tay (Điểm 0)
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    
    coords = []
    for lm in hand_landmarks:
        coords.append([lm.x - wrist_x, lm.y - wrist_y])
        
    # 2. Đối xứng gương nếu là tay trái (Đồng bộ hóa về form tay phải)
    if handedness_label == "Left":
        for i in range(21):
            coords[i][0] = -coords[i][0]
            
    # 3. Co giãn tỷ lệ về khoảng [-1, 1]
    flattened = [val for pt in coords for val in pt]
    max_val = max(abs(v) for v in flattened)
    if max_val == 0: max_val = 1.0
    
    normalized_vector = []
    for pt in coords:
        normalized_vector.extend([pt[0] / max_val, pt[1] / max_val])
        
    return normalized_vector

dataset_path = "./Dataset"
set_types = ["train", "test"]
data_list = []

with HandLandmarker.create_from_options(options) as landmarker:
    for set_type in set_types:
        set_path = os.path.join(dataset_path, set_type)
        if not os.path.exists(set_path): continue
            
        print(f"\n--- Đang trích xuất đặc trưng tập: {set_type} ---")
        for label in sorted(os.listdir(set_path)):
            label_path = os.path.join(set_path, label)
            if not os.path.isdir(label_path): continue
                
            image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in tqdm(image_files, desc=f"Nhãn {label}"):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if image is None: continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                detection_result = landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    # Lấy bàn tay chủ đạo đầu tiên
                    hand_landmarks = detection_result.hand_landmarks[0]
                    hand_label = detection_result.handedness[0][0].category_name # "Left" hoặc "Right"
                    
                    # Áp dụng bộ lọc chuẩn hóa toán học
                    features = extract_and_normalize(hand_landmarks, hand_label)
                    
                    row = {'path': img_path, 'set_type': set_type, 'label': label}
                    for i in range(21):
                        row[f'x{i}'] = features[i*2]
                        row[f'y{i}'] = features[i*2 + 1]
                    data_list.append(row)

if data_list:
    df = pd.DataFrame(data_list)
    df.to_csv("hand_landmarks_raw_v1.csv", index=False, encoding='utf-8')
    print(f"\n[THÀNH CÔNG] Đã lưu dữ liệu chuẩn hóa của {len(df)} ảnh.")