# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np


def letterbox(image, target_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to target_shape, returns resized image."""
    height, width = image.shape[:2]

    # Calculate scale and new size
    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_size = (int(width * scale), int(height * scale))

    # Resize the image
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    dw, dh = (target_shape[1] - new_size[0]) / 2, (target_shape[0] - new_size[1]) / 2
    top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
    left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))

    # Apply padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, scale, (dw, dh)


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape."""
    scale = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    dw, dh = (img1_shape[1] - img0_shape[1] * scale) / 2, (img1_shape[0] - img0_shape[0] * scale) / 2

    boxes[..., [0, 2]] -= dw  # x padding
    boxes[..., [1, 3]] -= dh  # y padding
    boxes[..., :4] /= scale

    clip_boxes(boxes, img0_shape)
    return boxes


def scale_landmarks(img1_shape, landmarks, img0_shape):
    """Rescales landmarks from img1_shape to img0_shape."""
    scale = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    dw, dh = (img1_shape[1] - img0_shape[1] * scale) / 2, (img1_shape[0] - img0_shape[0] * scale) / 2

    landmarks[:, 0::2] = (landmarks[:, 0::2] - dw) / scale  # x coordinates
    landmarks[:, 1::2] = (landmarks[:, 1::2] - dh) / scale  # y coordinates

    return landmarks


def draw_detections(image: np.ndarray, box: np.ndarray, score: float, landmarks: np.ndarray) -> None:
    """
    Draw face bounding box and landmarks on image.

    Args:
        image: Input image
        box: Bounding box [x1, y1, x2, y2]
        score: Confidence score
        landmarks: Facial landmarks [10] (5 points * 2 coords)
    """
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw confidence score
    label = f"{score:.2f}"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    # Draw facial landmarks (5 points)
    landmarks = landmarks.reshape(5, 2).astype(int)
    for i, (lx, ly) in enumerate(landmarks):
        if i < 2:
            landmark_color = (0, 0, 255)  # Red for eyes
        elif i == 2:
            landmark_color = (255, 0, 0)  # Blue for nose
        else:
            landmark_color = (0, 255, 0)  # Green for mouth
        cv2.circle(image, (lx, ly), 3, landmark_color, -1)
