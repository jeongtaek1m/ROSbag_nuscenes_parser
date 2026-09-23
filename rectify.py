"""Undistort camera images to the pinhole model NuScenes assumes.

NuScenes stores a 3x3 `camera_intrinsic` and no distortion coefficients, so
every consumer — the devkit's `view_points`, mmdetection3d, BEVFusion — projects
with a plain pinhole K. Our cameras are OpenCV fisheye (4 coefficients) or
plumb_bob (5), so the images are remapped here and the *new* K is what goes into
calibrated_sensor.json. The camera frame itself is unchanged (no rectifying
rotation), so the extrinsic stays as calibrated.
"""
from __future__ import annotations

import cv2
import numpy as np


class Rectifier:
    """Remap one camera's JPEGs to a distortion-free pinhole image of the same size.

    `balance` (0..1) sets the new focal length: 0 keeps only pixels that are
    valid across the whole output (no black border, the periphery is cropped),
    1 keeps the whole source field of view (black corners, stretched edges).
    A camera whose distortion is all zeros is passed through byte for byte.

    Instances are shared by the staging worker threads; __call__ only reads the
    precomputed maps, and OpenCV releases the GIL, so the work runs in parallel.
    """

    def __init__(self, K, D, model: str, size_wh: tuple[int, int],
                 balance: float = 0.0, jpeg_quality: int = 90):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(D, dtype=np.float64).reshape(-1)
        w, h = int(size_wh[0]), int(size_wh[1])
        self.model = model
        self.size_wh = (w, h)
        self.balance = float(balance)
        self.jpeg_quality = int(jpeg_quality)
        self.K = K
        self.D = D
        self._maps = None
        if not np.any(D):
            self.K_new = K
            return
        if model == "fisheye":
            d4 = D[:4].reshape(4, 1)
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, d4, (w, h), np.eye(3), balance=self.balance, new_size=(w, h))
            self._maps = cv2.fisheye.initUndistortRectifyMap(
                K, d4, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
        elif model == "pinhole":
            K_new, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), self.balance, (w, h))
            self._maps = cv2.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv2.CV_16SC2)
        else:
            raise ValueError(f"cannot rectify distortion model {model!r} "
                             f"({len(D)} coefficients; expected 4 or 5)")
        self.K_new = np.asarray(K_new, dtype=np.float64)

    @property
    def passthrough(self) -> bool:
        return self._maps is None

    def __call__(self, jpeg: bytes) -> bytes:
        if self._maps is None:
            return jpeg
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("undecodable JPEG")
        if (img.shape[1], img.shape[0]) != self.size_wh:
            raise ValueError(f"image is {img.shape[1]}x{img.shape[0]}, rectifier was "
                             f"built for {self.size_wh[0]}x{self.size_wh[1]}")
        out = cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return enc.tobytes()

    def describe(self) -> dict:
        """JSON-friendly record of what was done, for <log>.import.json."""
        return {
            "model": self.model,
            "size_wh": list(self.size_wh),
            "K": self.K.tolist(),
            "distortion": self.D.tolist(),
            "K_new": self.K_new.tolist(),
            "balance": self.balance,
            "passthrough": self.passthrough,
            "jpeg_quality": None if self.passthrough else self.jpeg_quality,
        }
