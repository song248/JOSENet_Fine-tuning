import os
import csv
import torch
import cv2
import numpy as np
import yaml
from types import SimpleNamespace
from sklearn.metrics import confusion_matrix, classification_report

from architectures import FGN_RGB, FGN_FLOW, FGN_MERGE_CLASSIFY, FGN
from preprocessing import generate_flow
from roi import roi_video
from tqdm import tqdm

# 경로 설정
ROOT_DIR = "test_video"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
LABEL_DIR = os.path.join(ROOT_DIR, "label")
PREDICT_DIR = os.path.join(ROOT_DIR, "predict")

MODEL_PATH = "models/primary/model_primary.pt"
CONFIG_PATH = "config/primary.yaml"
CLIP_FRAMES = 16
STRIDE = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 설정 불러오기
def load_args(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config)
    args.clip_frames = CLIP_FRAMES
    args.interval = STRIDE
    args.device = DEVICE
    return args

# 비디오 전처리
def normalize_segment(rgb_segment, flow_segment):
    rgb_segment = roi_video(rgb_segment, flow_segment)
    rgb = np.array(rgb_segment)
    flow = np.array(flow_segment)
    mean_rgb = np.mean(rgb)
    std_rgb = np.std(rgb)
    mean_flow = np.mean(flow)
    std_flow = np.std(flow)
    rgb = (rgb - mean_rgb) / (std_rgb + 1e-6)
    flow = (flow - mean_flow) / (std_flow + 1e-6)
    return rgb, flow

def segment_tensor(rgb_segment, flow_segment):
    rgb, flow = normalize_segment(rgb_segment, flow_segment)
    rgb_tensor = torch.FloatTensor(np.transpose(rgb, (3, 0, 1, 2)))
    flow_tensor = torch.FloatTensor(np.transpose(flow, (3, 0, 1, 2)))
    return rgb_tensor.unsqueeze(0), flow_tensor.unsqueeze(0)

def extract_segments(video_path, clip_len=16, stride=4):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()

    segments = []
    for start in range(0, len(frames) - clip_len + 1, stride):
        clip = frames[start:start + clip_len]
        rgb_segment = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]
        flow_segment = []
        for i in range(len(clip) - 1):
            flow_segment.append(generate_flow(clip[i], clip[i + 1]))
        flow_segment.append(np.zeros((IMG_SIZE, IMG_SIZE, 2)))
        segments.append((rgb_segment, flow_segment))
    return segments, len(frames)

# 모델 로딩
def load_model(model_path, args):
    model_rgb = FGN_RGB(args)
    model_flow = FGN_FLOW(args)
    model_merge = FGN_MERGE_CLASSIFY(args)
    model = FGN(model_rgb, model_flow, model_merge)
    model = torch.nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    return model

# 예측 및 CSV 저장
def predict_and_save(video_path, model, args):
    segments, total_frames = extract_segments(video_path, clip_len=args.clip_frames, stride=args.interval)
    frame_results = [0] * total_frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(PREDICT_DIR, f"{video_name}.csv")

    for i, (rgb, flow) in enumerate(segments):
        rgb_input, flow_input = segment_tensor(rgb, flow)
        rgb_input = rgb_input.to(args.device)
        flow_input = flow_input.to(args.device)

        with torch.no_grad():
            logits = torch.squeeze(model(rgb_input, flow_input))
            prob = torch.sigmoid(logits).item()
            label = 1 if prob >= 0.5 else 0
            start_frame = i * args.interval
            end_frame = start_frame + args.clip_frames
            for f in range(start_frame, min(end_frame, total_frames)):
                frame_results[f] = label

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "anomaly"])
        for i, v in enumerate(frame_results):
            writer.writerow([i, v])

# 평가 지표 출력
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

def run_all():
    os.makedirs(PREDICT_DIR, exist_ok=True)
    args = load_args(CONFIG_PATH)
    model = load_model(MODEL_PATH, args)

    video_files = [f for f in os.listdir(ROOT_DIR) if f.lower().endswith(VIDEO_EXTS)]

    print("🔎 영상 처리 중...")
    for fname in tqdm(video_files, desc="Predicting videos", unit="video"):
        video_path = os.path.join(ROOT_DIR, fname)
        predict_and_save(video_path, model, args)

    print("\n📊 예측 평가 중...")
    evaluate_predictions()

if __name__ == "__main__":
    run_all()
