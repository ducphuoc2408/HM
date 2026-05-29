import os
import pandas as pd

# Lấy đường dẫn tuyệt đối của thư mục chứa file code hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "Dataset") 
csv_path = os.path.join(current_dir, "hand_landmarks_raw.csv")

# 1. Kiểm tra xem file CSV đã tồn tại chưa
if not os.path.exists(csv_path):
    print(f"[LỖI] Không tìm thấy file CSV tại: {csv_path}")
    print("Vui lòng chạy file trích xuất dữ liệu trước!")
    exit()

# 2. Đọc file CSV và lấy danh sách các đường dẫn ảnh đã trích xuất thành công
print("Đang đọc dữ liệu từ file CSV...")
df = pd.read_csv(csv_path)

# Chuyển đổi toàn bộ đường dẫn trong CSV về dạng chuẩn của hệ điều hành để đối chiếu chính xác
# (Ví dụ: Đổi dấu gạch chéo xuôi/ngược cho đồng nhất)
csv_images_set = set(os.path.abspath(p) for p in df['path'])

set_types = ['train', 'test']
missing_images = []
total_disk_images = 0

print("Đang quét thư mục Dataset để đối chiếu...")

# 3. Quét toàn bộ ảnh thực tế trên ổ đĩa để so sánh với file CSV
if os.path.exists(dataset_path):
    for set_type in set_types:
        set_path = os.path.join(dataset_path, set_type)
        if not os.path.exists(set_path): continue
            
        for label in sorted(os.listdir(set_path)):
            label_path = os.path.join(set_path, label)
            if not os.path.isdir(label_path): continue
                
            images = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in images:
                img_path = os.path.abspath(os.path.join(label_path, img_name))
                total_disk_images += 1
                
                # Nếu ảnh có trên ổ đĩa nhưng KHÔNG CÓ trong file CSV -> Ảnh bị AI bỏ sót
                if img_path not in csv_images_set:
                    missing_images.append({
                        'set_type': set_type,
                        'label': label,
                        'image_name': img_name,
                        'full_path': img_path
                    })

# 4. In báo cáo tổng kết ra màn hình
print("="*65)
print(f"{'BÁO CÁO ĐỐI CHIẾU DỮ LIỆU FILE CSV':^65}")
print("="*65)
print(f"Tổng số ảnh thực tế có trong thư mục Dataset: {total_disk_images} ảnh.")
print(f"Tổng số ảnh trích xuất thành công trong CSV  : {len(df)} ảnh.")
print(f"Số lượng ảnh bị AI bỏ sót (không có trong CSV): {len(missing_images)} ảnh.")
print("-"*65)

# 5. Xuất danh sách ảnh bị thiếu ra file text để bạn kiểm tra từng ảnh một
if missing_images:
    report_file = "danh_sach_anh_bi_bo_sot.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== DANH SÁCH CÁC ẢNH BỊ AI BỎ SÓT (KHÔNG CÓ TRONG FILE CSV) ===\n\n")
        for i, item in enumerate(missing_images, 1):
            f.write(f"{i}. Tập: {item['set_type']} | Nhãn: {item['label']} | Tên ảnh: {item['image_name']}\n")
            f.write(f"   Đường dẫn: {item['full_path']}\n\n")
            
    print(f"[LƯU Ý] Đã lưu chi tiết đường dẫn của {len(missing_images)} ảnh bị thiếu vào file:")
    print(f" -> {os.path.abspath(report_file)}")
    print("Bạn có thể mở file .txt này lên để xem những ảnh nào đang bị lỗi.")
else:
    print("[HOÀN HẢO] File CSV có đầy đủ 100% dữ liệu, không bỏ sót bất kỳ ảnh nào!")
print("="*65)

import pandas as pd
df = pd.read_csv("hand_landmarks_raw.csv")
# In ra số lượng ảnh còn lại của từng nhãn trong tập train và test
print(df.groupby(['set_type', 'label']).size())