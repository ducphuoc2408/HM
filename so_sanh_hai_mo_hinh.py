import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib


def load_test_data(csv_path="hand_landmarks_raw.csv"):
    df = pd.read_csv(csv_path)
    # Lấy tập test chuyên dụng, nếu không có thì lấy toàn bộ dữ liệu để đánh giá
    if 'set_type' in df.columns:
        test_df = df[df.set_type == 'test']
        if test_df.empty:
            print("Warning: 'set_type' exists but no 'test' samples found. Using validation split logic or entire data.")
            test_df = df
    else:
        test_df = df

    feature_cols = [c for c in df.columns if c.startswith('x') or c.startswith('y')]
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['label'].values
    return X_test, y_test


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


def plot_metrics_comparison(ann_metrics, svm_metrics, out_dir):
    """Vẽ biểu đồ cột so sánh tổng quan các chỉ số giữa 2 mô hình"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, ann_metrics, width, label='ANN', color='#1f77b4')
    plt.bar(x + width/2, svm_metrics, width, label='SVM', color='#ff7f0e')

    plt.ylabel('Điểm số (0.0 - 1.0)')
    plt.title('So sánh hiệu năng tổng quan: ANN vs SVM')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hiển thị số trên đầu cột
    for i in range(len(metrics_names)):
        plt.text(i - width/2, ann_metrics[i] + 0.02, f"{ann_metrics[i]:.3f}", ha='center', fontsize=9)
        plt.text(i + width/2, svm_metrics[i] + 0.02, f"{svm_metrics[i]:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_metrics_comparison.png'))
    print(f" Saved metrics comparison chart to: {out_dir}/model_metrics_comparison.png")


def plot_confusion_matrices(y_true, ann_pred, svm_pred, classes, out_dir):
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) cạnh nhau để đối chiếu trực quan"""
    ann_cm = confusion_matrix(y_true, ann_pred)
    svm_cm = confusion_matrix(y_true, svm_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Ma trận của ANN
    sns.heatmap(ann_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title('Ma trận nhầm lẫn - ANN')
    axes[0].set_xlabel('Nhãn Dự Đoán')
    axes[0].set_ylabel('Nhãn Thực Tế')

    # Ma trận của SVM
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title('Ma trận nhầm lẫn - SVM')
    axes[1].set_xlabel('Nhãn Dự Đoán')
    axes[1].set_ylabel('Nhãn Thực Tế')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrices_comparison.png'))
    print(f" Saved confusion matrices chart to: {out_dir}/confusion_matrices_comparison.png")


def main(args):
    # 1. Load data
    print("Loading test data...")
    X_test, y_test = load_test_data(args.csv)
    
    # 2. Load các artifact (Scaler, LabelEncoder, Models)
    print("Loading models and preprocessing pipelines...")
    mapping_path = os.path.join(args.model_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(args.model_dir, 'scaler.joblib')
    ann_path = os.path.join(args.model_dir, 'ann_model.joblib')
    svm_path = os.path.join(args.model_dir, 'svm_model.joblib')

    with open(mapping_path, 'rb') as f:
        le = pickle.load(f)
    scaler = joblib.load(scaler_path)
    ann_clf = joblib.load(ann_path)
    svm_clf = joblib.load(svm_path)

    # Encode nhãn thực tế sang số
    y_test_enc = le.transform(y_test)

    # 3. Thực hiện tiền xử lý dữ liệu kiểm thử đồng bộ
    X_test_norm = normalize_landmarks(X_test)
    X_test_scaled = scaler.transform(X_test_norm)

    # 4. Dự đoán từ 2 mô hình
    print("Evaluating models...")
    ann_pred = ann_clf.predict(X_test_scaled)
    svm_pred = svm_clf.predict(X_test_scaled)

    # 5. Tính toán các chỉ số chi tiết
    ann_acc = accuracy_score(y_test_enc, ann_pred)
    svm_acc = accuracy_score(y_test_enc, svm_pred)

    # Tính toán Weighted Avg cho Precision, Recall, F1
    ann_p, ann_r, ann_f, _ = precision_recall_fscore_support(y_test_enc, ann_pred, average='weighted')
    svm_p, svm_r, svm_f, _ = precision_recall_fscore_support(y_test_enc, svm_pred, average='weighted')

    # In báo cáo chi tiết ra Terminal
    print("\n=================== ANN CLASSIFICATION REPORT ===================")
    print(classification_report(y_test_enc, ann_pred, target_names=le.classes_))
    
    print("\n=================== SVM CLASSIFICATION REPORT ===================")
    print(classification_report(y_test_enc, svm_pred, target_names=le.classes_))

    print("\n======================= SUMMARY COMPARISON =======================")
    print(f"{'Metric':<15} | {'ANN Model':<15} | {'SVM Model':<15}")
    print("-" * 53)
    print(f"{'Accuracy':<15} | {ann_acc:<15.4f} | {svm_acc:<15.4f}")
    print(f"{'Precision':<15} | {ann_p:<15.4f} | {svm_p:<15.4f}")
    print(f"{'Recall':<15} | {ann_r:<15.4f} | {svm_r:<15.4f}")
    print(f"{'F1-Score':<15} | {ann_f:<15.4f} | {svm_f:<15.4f}")
    print("==================================================================")

    # 6. Vẽ và lưu biểu đồ
    os.makedirs(args.out_dir, exist_ok=True)
    
    ann_metrics_list = [ann_acc, ann_p, ann_r, ann_f]
    svm_metrics_list = [svm_acc, svm_p, svm_r, svm_f]
    
    plot_metrics_comparison(ann_metrics_list, svm_metrics_list, args.out_dir)
    plot_confusion_matrices(y_test_enc, ann_pred, svm_pred, le.classes_, args.out_dir)
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and Compare ANN vs SVM models')
    parser.add_argument('--csv', type=str, default='hand_landmarks_raw.csv', help='Path to test or raw CSV')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory where models are saved')
    parser.add_argument('--out-dir', type=str, default='evaluation_results', help='Directory to save charts')
    parser.add_argument('--show', action='store_true', help='Show plots interactively during execution')
    args = parser.parse_args()
    main(args)