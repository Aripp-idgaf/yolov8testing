# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
import torchvision


class YOLOv8Face:
    """YOLOv8-Face ONNX inference class."""

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        nms_mode: str = "torchvision",
    ) -> None:
        """Initialize YOLOv8-Face ONNX model.

        Args:
            model_path (str): Path to ONNX model file
            conf_thres (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold for NMS. Defaults to 0.45.
            max_det (int, optional): Maximum number of detections. Defaults to 300.
            nms_mode (str, optional): NMS calculation method ('torchvision', 'numpy'). Defaults to 'torchvision'.
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        # Initialize model (sets self.img_size from ONNX input shape)
        self._initialize_model(model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the given image and return predictions.

        Args:
            image (np.ndarray): Input image (preprocessed/resized).

        Returns:
            Tuple: (boxes, scores, landmarks) where:
                - boxes: [N, 4] bounding boxes in xyxy format
                - scores: [N] confidence scores
                - landmarks: [N, 10] facial landmarks (5 points * 2 coords)
        """
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
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get input shape from model (e.g., [1, 3, 640, 640])
            input_shape = self.session.get_inputs()[0].shape
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
        """Preprocess image for inference.

        Args:
            img (np.ndarray): Input image (BGR format, already resized with letterbox)

        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Convert BGR to RGB
        img = img[:, :, ::-1]

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img

    def postprocess(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Postprocess model predictions.

        Args:
            predictions (List[np.ndarray]): Raw model outputs (3 feature maps)

        Returns:
            np.ndarray: Filtered detections [x1, y1, x2, y2, conf, landmarks...]
        """
        # YOLOv8-Face outputs 3 feature maps with Pose head
        # Each output: (1, 80, H, W) where 80 = 64 (bbox DFL) + 1 (class) + 15 (5 keypoints * 3)

        boxes_list = []
        scores_list = []
        landmarks_list = []

        strides = [8, 16, 32]  # YOLOv8 strides for 640x640 input

        for pred, stride in zip(predictions, strides):
            # pred shape: (1, 80, H, W)
            batch_size, channels, height, width = pred.shape

            # Reshape: (1, 80, H, W) -> (1, 80, H*W) -> (1, H*W, 80) -> (H*W, 80)
            pred = pred.reshape(batch_size, channels, -1).transpose(0, 2, 1)[0]

            # Create grid with 0.5 offset (matching PyTorch's make_anchors)
            grid_y, grid_x = np.meshgrid(np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij")
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            # Extract components
            bbox_pred = pred[:, :64]  # DFL bbox prediction
            cls_conf = pred[:, 64]  # Class confidence
            kpt_pred = pred[:, 65:]  # 15 keypoint values (5 points * 3: x, y, visibility)

            # Decode bounding boxes from DFL
            bbox_pred = bbox_pred.reshape(-1, 4, 16)
            bbox_dist = self.softmax(bbox_pred, axis=-1) @ np.arange(16)

            # Convert distances to xyxy format
            x1 = (grid_x - bbox_dist[:, 0]) * stride
            y1 = (grid_y - bbox_dist[:, 1]) * stride
            x2 = (grid_x + bbox_dist[:, 2]) * stride
            y2 = (grid_y + bbox_dist[:, 3]) * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode keypoints: kpt = (kpt * 2.0 + grid) * stride
            kpt_grid_y, kpt_grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            kpt_grid_x = kpt_grid_x.flatten()
            kpt_grid_y = kpt_grid_y.flatten()

            kpt_pred = kpt_pred.reshape(-1, 5, 3)
            kpt_x = (kpt_pred[:, :, 0] * 2.0 + kpt_grid_x[:, None]) * stride
            kpt_y = (kpt_pred[:, :, 1] * 2.0 + kpt_grid_y[:, None]) * stride
            landmarks = np.stack([kpt_x, kpt_y], axis=-1).reshape(-1, 10)

            # Apply sigmoid to class confidence
            scores = 1 / (1 + np.exp(-cls_conf))

            boxes_list.append(boxes)
            scores_list.append(scores)
            landmarks_list.append(landmarks)

        # Concatenate all predictions
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        landmarks = np.concatenate(landmarks_list, axis=0)

        # Filter by confidence
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(boxes) == 0:
            return np.array([])

        # Apply NMS
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

        # Filter detections and limit to max_det
        indices = indices[: self.max_det]
        boxes = boxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Combine results
        detections = np.concatenate([boxes, scores[:, None], landmarks], axis=1)

        return detections

    @staticmethod
    def softmax(x, axis=-1):
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """Convert bounding box format from xywh to xyxy.

        Args:
            x (np.ndarray): Boxes in [x, y, w, h] format

        Returns:
            np.ndarray: Boxes in [x1, y1, x2, y2] format
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression (NumPy implementation).

        Args:
            boxes (np.ndarray): Bounding boxes [x1, y1, x2, y2]
            scores (np.ndarray): Confidence scores
            iou_threshold (float): IoU threshold

        Returns:
            List[int]: Indices of boxes to keep
        """
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
        """Run face detection on image.

        Args:
            img (np.ndarray): Input image (BGR format, already resized with letterbox)

        Returns:
            np.ndarray: Detections [x1, y1, x2, y2, conf, landmarks...]
        """
        # Preprocess
        input_tensor = self.preprocess(img)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs)

        return detections
