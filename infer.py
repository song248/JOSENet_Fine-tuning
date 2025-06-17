import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import yaml
from types import SimpleNamespace

from architectures import FGN_RGB, FGN_FLOW, FGN_MERGE_CLASSIFY, FGN
from preprocessing import generate_flow
from roi import roi_video  # ROI 적용
import albumentations as A

VIDEO_PATH = "test_02-1.mp4"
MODEL_PATH = "models/primary/model_primary.pt"
CONFIG_PATH = "config/primary.yaml"
CLIP_FRAMES = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_args(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config)
    args.clip_frames = CLIP_FRAMES
    args.device = DEVICE
    return args

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

def load_model(model_path, args):
    model_rgb = FGN_RGB(args)
    model_flow = FGN_FLOW(args)
    model_merge = FGN_MERGE_CLASSIFY(args)
    model = FGN(model_rgb, model_flow, model_merge)
    model = torch.nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    return model

def predict(video_path):
    args = load_args(CONFIG_PATH)
    model = load_model(MODEL_PATH, args)
    segments, total_frames = extract_segments(video_path, clip_len=args.clip_frames, stride=args.interval)

    print(f"\n🎞️ 영상: {os.path.basename(video_path)}")
    print(f"📌 총 프레임 수: {total_frames}")
    print(f"📦 생성된 세그먼트 수: {len(segments)} (stride={args.interval}, clip_frames={args.clip_frames})\n")

    probs = []
    for i, (rgb, flow) in enumerate(segments):
        rgb_input, flow_input = segment_tensor(rgb, flow)
        rgb_input = rgb_input.to(args.device)
        flow_input = flow_input.to(args.device)

        with torch.no_grad():
            logits = torch.squeeze(model(rgb_input, flow_input))
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            print(f"📦 Segment {i+1:02d}: 이상 확률 {prob * 100:.2f}% {'🔥' if prob >= 0.5 else '✅'}")

    final = max(probs)
    print(f"\n📊 최종 판단 (최댓값 기준): {'Fight 🔥' if final >= 0.5 else 'NonFight ✅'}")

if __name__ == "__main__":
    predict(VIDEO_PATH)
