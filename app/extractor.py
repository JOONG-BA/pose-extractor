from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class FramePoseRow:
    frame_index: int
    landmark_index: int
    x: float
    y: float
    z: float
    visibility: float
    world_x: float
    world_y: float
    world_z: float


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

    def load_gif_frames(self, gif_path: str) -> List[np.ndarray]:
        frames = imageio.mimread(gif_path)
        rgb_frames: List[np.ndarray] = []

        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            rgb_frames.append(arr.astype(np.uint8))

        return rgb_frames

    def detect_on_frame(self, rgb_frame: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)

    def extract_from_gif(self, gif_path: str) -> pd.DataFrame:
        rows: list[dict] = []
        frames = self.load_gif_frames(gif_path)

        for frame_idx, rgb_frame in enumerate(frames):
            result = self.detect_on_frame(rgb_frame)

            if not result.pose_landmarks:
                continue

            image_landmarks = result.pose_landmarks[0]
            world_landmarks = result.pose_world_landmarks[0]

            for lm_idx, (img_lm, world_lm) in enumerate(zip(image_landmarks, world_landmarks)):
                rows.append(
                    {
                        "frame_index": frame_idx,
                        "landmark_index": lm_idx,
                        "x": float(img_lm.x),
                        "y": float(img_lm.y),
                        "z": float(img_lm.z),
                        "visibility": float(img_lm.visibility),
                        "world_x": float(world_lm.x),
                        "world_y": float(world_lm.y),
                        "world_z": float(world_lm.z),
                    }
                )

        return pd.DataFrame(rows)

    def draw_landmarks(self, rgb_frame: np.ndarray, detection_result) -> np.ndarray:
        annotated = rgb_frame.copy()

        if not detection_result.pose_landmarks:
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        pose_landmarks = detection_result.pose_landmarks[0]
        h, w, _ = annotated.shape

        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    def export_preview_frames(self, gif_path: str, out_dir: str, max_frames: int = 10) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        frames = self.load_gif_frames(gif_path)

        for idx, rgb_frame in enumerate(frames[:max_frames]):
            result = self.detect_on_frame(rgb_frame)
            preview = self.draw_landmarks(rgb_frame, result)
            cv2.imwrite(str(out_path / f"frame_{idx:04d}.png"), preview)