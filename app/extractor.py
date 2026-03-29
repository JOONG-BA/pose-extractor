from __future__ import annotations

from pathlib import Path
from typing import Iterable
    
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseExtractor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.landmarker = self._create_landmarker()

    def _create_landmarker(self) -> vision.PoseLandmarker:
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def load_image(self, image_path: str) -> np.ndarray:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def detect_on_image(self, rgb_image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    def extract_from_image(self, image_path: str, image_index: int = 0) -> list[dict]:
        rgb_image = self.load_image(image_path)
        result = self.detect_on_image(rgb_image)
        rows: list[dict] = []

        if not result.pose_landmarks:
            return rows

        image_landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0]
        image_name = Path(image_path).name

        for landmark_index, (img_lm, world_lm) in enumerate(zip(image_landmarks, world_landmarks)):
            rows.append(
                {
                    "image_index": image_index,
                    "image_name": image_name,
                    "landmark_index": landmark_index,
                    "x": float(img_lm.x),
                    "y": float(img_lm.y),
                    "z": float(img_lm.z),
                    "visibility": float(img_lm.visibility),
                    "world_x": float(world_lm.x),
                    "world_y": float(world_lm.y),
                    "world_z": float(world_lm.z),
                }
            )

        return rows

    def extract_from_images(self, image_paths: Iterable[str]) -> pd.DataFrame:
        rows: list[dict] = []

        for image_index, image_path in enumerate(image_paths):
            rows.extend(self.extract_from_image(image_path, image_index=image_index))

        return pd.DataFrame(rows)

    def draw_landmarks(self, rgb_image: np.ndarray, detection_result) -> np.ndarray:
        annotated = rgb_image.copy()

        if not detection_result.pose_landmarks:
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        pose_landmarks = detection_result.pose_landmarks[0]
        h, w, _ = annotated.shape

        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    def export_preview_images(self, image_paths: Iterable[str], out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            rgb_image = self.load_image(image_path)
            result = self.detect_on_image(rgb_image)
            preview = self.draw_landmarks(rgb_image, result)
            stem = Path(image_path).stem
            cv2.imwrite(str(out_path / f"{stem}_preview.png"), preview)
