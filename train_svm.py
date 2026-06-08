import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC  # Thay thế MLPClassifier bằng Support Vector Classification
from sklearn.model_selection import train_test_split
import joblib


def load_data(csv_path="hand_landmarks_raw.csv"):
    df = pd.read_csv(csv_path)
    if 'set_type' in df.columns:
        train_df = df[df.set_type == 'train']
        test_df = df[df.set_type == 'test']
    else:
        train_df = df
        test_df = pd.DataFrame()

    feature_cols = [c for c in df.columns if c.startswith('x') or c.startswith('y')]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['label'].values

    if not test_df.empty:
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df['label'].values
    else:
        X_test = None
        y_test = None

    return X_train, y_train, X_test, y_test


def normalize_landmarks(X):
    # X shape (n_samples, 42) -> normalize per sample: subtract wrist (x0,y0), scale by max distance
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


def main(args):
    X_train, y_train, X_test, y_test = load_data(args.csv)

    # Label encode
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    if X_test is not None and y_test is not None:
        y_test_enc = le.transform(y_test)

    # If no explicit test set, split from train
    if X_test is None or y_test is None:
        X_train_arr, X_val, y_train_arr, y_val = train_test_split(
            X_train, y_train_enc, test_size=0.15, random_state=42, stratify=y_train_enc)
    else:
        X_train_arr, y_train_arr = X_train, y_train_enc
        X_val, y_val = X_test, y_test_enc

    # normalize landmarks (translation + scale invariant)
    X_train_arr = normalize_landmarks(X_train_arr)
    if X_val is not None:
        X_val = normalize_landmarks(X_val)

    # standardize features (Rất quan trọng đối với SVM dựa trên khoảng cách)
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train_arr)
    if X_val is not None:
        X_val = scaler.transform(X_val)

    # Khởi tạo mô hình SVM thay vì MLP
    # kernel='rbf' (Radial Basis Function) hoạt động rất tốt với dữ liệu landmark hình học phi tuyến tính
    # probability=True cho phép mô hình xuất ra xác suất (confidence score) giống như Softmax của ANN
    print(f"Starting SVM training with C={args.C}, gamma={args.gamma}...")
    clf = SVC(kernel='rbf', C=args.C, gamma=args.gamma, probability=True, random_state=42)

    clf.fit(X_train_arr, y_train_arr)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, 'svm_model.joblib')
    mapping_path = os.path.join(args.out_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(args.out_dir, 'scaler.joblib')

    # Lưu model, scaler và label encoder tương thích với hệ thống cũ
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    with open(mapping_path, 'wb') as f:
        pickle.dump(le, f)

    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {mapping_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Đánh giá độ chính xác
    if X_val is not None and y_val is not None:
        y_pred = clf.predict(X_val)
        print("\n--- Classification Report (SVM) ---")
        print(classification_report(y_val, y_pred, target_names=le.classes_))
    else:
        print("No validation/test set provided.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SVM on hand landmarks CSV')
    parser.add_argument('--csv', type=str, default='hand_landmarks_raw.csv')
    parser.add_argument('--out-dir', type=str, default='models')
    
    # Loại bỏ --epochs và --batch-size vì SVM không huấn luyện theo dạng lặp batch.
    # Thay bằng hai siêu tham số quan trọng nhất của RBF SVM:
    parser.add_argument('--C', type=float, default=1.0, 
                        help='Penalty parameter C of the error term.')
    parser.add_argument('--gamma', type=str, default='scale', 
                        help="Kernel coefficient for 'rbf'. Can be 'scale', 'auto' or a float.")
    
    args = parser.parse_args()
    
    # Ép kiểu dữ liệu cho gamma nếu người dùng nhập số
    try:
        args.gamma = float(args.gamma)
    except ValueError:
        pass

    main(args)