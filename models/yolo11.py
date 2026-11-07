# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
import torchvision


class YOLO11Face:
    """YOLO11-Face ONNX inference class."""

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        nms_mode: str = "torchvision",
    ) -> None:
        """Initialize YOLO11-Face ONNX model."""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        # Initialize model (sets self.img_size from ONNX input shape)
        self._initialize_model(model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the given image and return predictions."""
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        detections = self.detect(image)

        if len(detections) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = detections[:, :4]
        scores = detections[:, 4]
        landmarks = detections[:, 5:]

        return boxes, scores, landmarks

    def _initialize_model(self, model_path: str) -> None:
        """Initialize the model from the given path."""
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get input shape from model (e.g., [1, 3, 640, 640])
            self.img_size = (640, 640)  # (H, W)

            # Get model metadata
            meta = self.session.get_modelmeta()
            if meta.custom_metadata_map:
                self.stride = int(meta.custom_metadata_map.get("stride", 32))
            else:
                self.stride = 32

        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}") from e

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def postprocess(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Postprocess model predictions (Updated to handle models without landmarks)."""
        pred = predictions[0][0]  
        pred = pred.transpose()   

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4]

        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        # Cek apakah model memiliki data landmarks (dimensi >= 20)
        if pred.shape[1] >= 20:
            raw_landmarks = pred[:, 5:20]
            pts = raw_landmarks.reshape(-1, 5, 3)
            landmarks = np.stack([pts[:, :, 0], pts[:, :, 1]], axis=-1).reshape(-1, 10)
        else:
            # Jika tidak ada landmarks, buat data dummy bernilai -1
            # Sesuaikan ukurannya dengan jumlah prediksi awal (pred.shape[0]) BUKAN boxes.shape[0]
            landmarks = np.full((pred.shape[0], 10), -1.0) 

        # Filter berdasarkan threshold confidence
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(boxes) == 0:
            return np.array([])

        # Terapkan Non-Maximum Suppression (NMS)
        if self.nms_mode == "torchvision":
            indices = torchvision.ops.nms(
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                self.iou_thres,
            ).numpy()
        else:
            indices = self.nms(boxes, scores, self.iou_thres)

        if len(indices) == 0:
            return np.array([])

        indices = indices[: self.max_det]
        boxes = boxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Gabungkan semua hasil kembali (box, score, landmarks)
        detections = np.concatenate([boxes, scores[:, None], landmarks], axis=1)

        return detections

    @staticmethod
    def softmax(x, axis=-1):
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """Convert bounding box format from xywh to xyxy."""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression (NumPy implementation)."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Run face detection on image."""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        detections = self.postprocess(outputs)
        return detections