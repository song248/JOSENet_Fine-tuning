import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 평가에 필요한 디렉토리 설정
ROOT_DIR = "test_video"
LABEL_DIR = os.path.join(ROOT_DIR, "label")
PREDICT_DIR = os.path.join(ROOT_DIR, "predict")

# 평가 함수
def evaluate_predictions():
    y_true_all, y_pred_all = [], []

    for fname in os.listdir(LABEL_DIR):
        if fname.endswith(".csv"):
            label_path = os.path.join(LABEL_DIR, fname)
            predict_path = os.path.join(PREDICT_DIR, fname)
            if not os.path.exists(predict_path):
                print(f"⚠️ 예측 결과 없음: {fname}")
                continue

            with open(label_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                label = [int(row[1]) for row in reader]

            with open(predict_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                pred = [int(row[1]) for row in reader]

            min_len = min(len(label), len(pred))
            y_true_all.extend(label[:min_len])
            y_pred_all.extend(pred[:min_len])

    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(cm)

    print("\n📋 Classification Report:")
    print(classification_report(y_true_all, y_pred_all, digits=4))

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-6)
    print(f"🧮 Specificity: {specificity:.4f}")

# 메인 실행
if __name__ == "__main__":
    evaluate_predictions()
