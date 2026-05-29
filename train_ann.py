
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


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


def build_model(input_dim, n_classes, hidden_sizes=(128, 64), dropout=0.3):
	model = Sequential()
	model.add(Dense(hidden_sizes[0], input_shape=(input_dim,), activation='relu'))
	model.add(Dropout(dropout))
	if len(hidden_sizes) > 1:
		model.add(Dense(hidden_sizes[1], activation='relu'))
		model.add(Dropout(dropout))
	model.add(Dense(n_classes, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


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

	# standardize features
	scaler = StandardScaler()
	X_train_arr = scaler.fit_transform(X_train_arr)
	if X_val is not None:
		X_val = scaler.transform(X_val)

	# build sklearn MLP
	clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
						early_stopping=True, max_iter=args.epochs, batch_size=args.batch_size,
						verbose=True, random_state=42)

	clf.fit(X_train_arr, y_train_arr)

	os.makedirs(args.out_dir, exist_ok=True)
	model_path = os.path.join(args.out_dir, 'ann_model.joblib')
	mapping_path = os.path.join(args.out_dir, 'label_encoder.pkl')

	# save model, scaler and label encoder
	joblib.dump(clf, model_path)
	scaler_path = os.path.join(args.out_dir, 'scaler.joblib')
	joblib.dump(scaler, scaler_path)
	with open(mapping_path, 'wb') as f:
		pickle.dump(le, f)

	print(f"Model saved to: {model_path}")
	print(f"Label encoder saved to: {mapping_path}")

	# evaluation
	if X_val is not None and y_val is not None:
		y_pred = clf.predict(X_val)
		print(classification_report(y_val, y_pred, target_names=le.classes_))
	else:
		print("No validation/test set provided.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train ANN on hand landmarks CSV')
	parser.add_argument('--csv', type=str, default='hand_landmarks_raw.csv')
	parser.add_argument('--out-dir', type=str, default='models')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch-size', type=int, default=32)
	args = parser.parse_args()
	main(args)

