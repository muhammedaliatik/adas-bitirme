"""Hibrit ADAS video pipeline.

YOLOPv2 şerit/sürüş alanı segmentasyonu ile YOLO11 nesne tespitini
aynı video karesinde birleştirir. Colab'a özel kod içermez; yerel ortamda
tekrar üretilebilir bir çalıştırma noktası sağlar.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class ClassInfo:
    name: str
    real_height_m: float
    color_bgr: Tuple[int, int, int]


CLASS_INFO: Dict[int, ClassInfo] = {
    0: ClassInfo("Yaya", 1.70, (0, 230, 255)),
    1: ClassInfo("Bisiklet", 1.20, (0, 255, 128)),
    2: ClassInfo("Araba", 1.50, (0, 0, 255)),
    3: ClassInfo("Motosiklet", 1.30, (203, 108, 240)),
    5: ClassInfo("Otobus", 3.20, (0, 165, 255)),
    7: ClassInfo("Kamyon", 3.50, (255, 200, 0)),
    9: ClassInfo("Trafik Isigi", 0.85, (180, 0, 180)),
    11: ClassInfo("Dur Levhasi", 0.80, (255, 100, 100)),
}


def distance_color(distance_m: float) -> Tuple[int, int, int]:
    if distance_m < 15.0:
        return (0, 0, 255)
    if distance_m <= 30.0:
        return (0, 140, 255)
    return (0, 230, 70)


def draw_badge(frame: np.ndarray, text: str, origin: Tuple[int, int],
               color: Tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness, pad = 0.48, 1, 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(frame.shape[1], x + tw + pad), min(frame.shape[0], y + baseline + pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


class HybridADAS:
    def __init__(self, yolopv2_path: str, yolo_path: str,
                 confidence: float = 0.35, device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolopv2 = self._load_yolopv2(yolopv2_path)
        self.yolo11 = YOLO(yolo_path)
        self.confidence = confidence

    def _load_yolopv2(self, path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLOPv2 ağırlığı bulunamadı: {model_path}")
        model = torch.jit.load(str(model_path), map_location=self.device).to(self.device)
        if self.device != "cpu":
            model.half()
        model.eval()
        return model

    def _lane_overlay(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = tensor.half() if self.device != "cpu" else tensor.float()
        tensor /= 255.0

        with torch.inference_mode():
            outputs = self.yolopv2(tensor)

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            return frame.copy()

        _, driving_area, lane_line = outputs[:3]
        lane_mask = torch.round(lane_line).squeeze().byte().cpu().numpy()
        lane_mask = cv2.resize(lane_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # YOLOPv2 sürüş alanı çıktısı çoğunlukla [batch, classes, h, w] biçimindedir.
        driving_tensor = driving_area[0] if isinstance(driving_area, (tuple, list)) else driving_area
        if driving_tensor.ndim == 4 and driving_tensor.shape[1] > 1:
            driving_tensor = driving_tensor[0, 1]
        elif driving_tensor.ndim == 3 and driving_tensor.shape[0] > 1:
            driving_tensor = driving_tensor[1]
        else:
            driving_tensor = driving_tensor.squeeze()
        driving_mask = torch.round(driving_tensor).byte().cpu().numpy()
        driving_mask = cv2.resize(driving_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        result = frame.copy()
        road = driving_mask == 1
        lane = lane_mask == 1
        result[road] = (result[road] * 0.70 + np.array([40, 200, 80]) * 0.30).astype(np.uint8)
        result[lane] = (0, 255, 255)
        return result

    def _draw_objects(self, frame: np.ndarray, focal_length_px: float) -> int:
        results = self.yolo11.predict(
            frame,
            conf=self.confidence,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
        )
        count = 0
        if not results:
            return count

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            info = CLASS_INFO.get(class_id)
            if info is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pixel_height = max(1, y2 - y1)
            distance_m = round((info.real_height_m * focal_length_px) / pixel_height, 1)
            risk_color = distance_color(distance_m)

            cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, 2)
            label = f"{info.name} | {distance_m:.1f}m"
            draw_badge(frame, label, (x1 + 3, max(18, y1 - 6)), info.color_bgr)
            count += 1
        return count

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        height = frame.shape[0]
        focal_length_px = (height / 2.0) / np.tan(np.radians(70.0) / 2.0)
        result = self._lane_overlay(frame)
        count = self._draw_objects(result, focal_length_px)
        return result, count


def process_video(input_path: str, output_path: str, yolopv2_path: str,
                  yolo_path: str, confidence: float, max_frames: int | None) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Giriş videosu bulunamadı: {input_file}")

    capture = cv2.VideoCapture(str(input_file))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {input_file}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Çıkış videosu oluşturulamadı: {output_path}")

    pipeline = HybridADAS(yolopv2_path, yolo_path, confidence)
    started = time.perf_counter()
    frame_index = 0
    detections = 0

    try:
        while frame_index < total:
            ok, frame = capture.read()
            if not ok:
                break
            result, count = pipeline.process_frame(frame)
            writer.write(result)
            detections += count
            frame_index += 1

            if frame_index % 30 == 0:
                elapsed = max(1e-6, time.perf_counter() - started)
                print(f"[{frame_index}/{total}] {frame_index / elapsed:.1f} FPS")
    finally:
        capture.release()
        writer.release()

    elapsed = max(1e-6, time.perf_counter() - started)
    print(f"Tamamlandı: {output_path}")
    print(f"Kare: {frame_index} | Ortalama FPS: {frame_index / elapsed:.1f} | Tespit: {detections}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOPv2 + YOLO11 hibrit ADAS pipeline")
    parser.add_argument("--input", default="test.mp4", help="Giriş videosu")
    parser.add_argument("--output", default="test_hibrit.mp4", help="Çıkış videosu")
    parser.add_argument("--yolopv2", default="yolopv2.pt", help="YOLOPv2 TorchScript ağırlığı")
    parser.add_argument("--yolo", default="yolo11s.pt", help="YOLO11 ağırlığı")
    parser.add_argument("--conf", type=float, default=0.35, help="Nesne güven eşiği")
    parser.add_argument("--max-frames", type=int, default=None, help="İsteğe bağlı kare limiti")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        args.input, args.output, args.yolopv2, args.yolo,
        args.conf, args.max_frames,
    )
