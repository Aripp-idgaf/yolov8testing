# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
"""
YOLOv8-Face ONNX Inference Script

Usage:
    python main.py --weights weights/yolov8n-face.onnx --source assets/test.jpg
    python main.py --weights weights/yolov8n-face.onnx --source 0  # webcam
    python main.py --weights weights/yolov8n-face.onnx --source video.mp4
"""

import argparse
import os

import cv2

from models import YOLOv8Face
from utils.general import draw_detections, letterbox, scale_boxes, scale_landmarks

VID_FORMATS = ["mp4", "avi", "mov", "mkv"]
IMG_FORMATS = ["jpg", "jpeg", "png", "bmp"]


def run_face_detection(
    weights: str,
    source: str,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    save_img: bool,
    view_img: bool,
) -> None:
    """
    Run face detection on image, video, or webcam.

    Args:
        weights: Path to ONNX model file
        source: Path to image/video file or webcam index
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
        save_img: Whether to save results
        view_img: Whether to display results
    """
    # Initialize model
    model = YOLOv8Face(weights, conf_thres, iou_thres, max_det=max_det)

    # Determine source type
    is_webcam = source.isdigit() or source == "0"
    if is_webcam:
        source_type = "webcam"
        cap = cv2.VideoCapture(int(source))
    else:
        ext = os.path.splitext(source)[1][1:].lower()
        if ext in VID_FORMATS:
            source_type = "video"
            cap = cv2.VideoCapture(source)
        elif ext in IMG_FORMATS:
            source_type = "image"
        else:
            raise ValueError(f"Unsupported format: {source}")

    # Auto-enable view_img for webcam
    if source_type == "webcam":
        view_img = True
        print("Webcam detected - auto-enabling display. Press 'q' to quit.")

    # Video writer for saving video/webcam output
    vid_writer = None
    if save_img and source_type in ["video", "webcam"]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_path = f"result_{os.path.basename(source)}.mp4" if source_type == "video" else "result_webcam.mp4"
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Process frames
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "video" else 0

    while True:
        if source_type == "image":
            frame = cv2.imread(source)
            if frame is None:
                print(f"Error: Could not read image {source}")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1

        # Preprocess
        img, scale, (dw, dh) = letterbox(frame, target_shape=model.img_size)

        # Inference
        boxes, scores, landmarks = model(img)

        # Scale coordinates back to original image
        if len(boxes) > 0:
            boxes = scale_boxes(model.img_size, boxes, frame.shape[:2])
            landmarks = scale_landmarks(model.img_size, landmarks, frame.shape[:2])

            # Draw detections
            for box, score, lm in zip(boxes, scores, landmarks):
                draw_detections(frame, box, score, lm)

        # Print status
        n_faces = len(boxes)
        if source_type == "webcam":
            status = f"Webcam (frame {frame_idx}): {n_faces} face{'s' * (n_faces != 1)}"
        elif source_type == "video":
            status = f"Video (frame {frame_idx}/{total_frames}): {n_faces} face{'s' * (n_faces != 1)}"
        else:
            status = f"Image {source}: {n_faces} face{'s' * (n_faces != 1)}"
        print(status)

        # Display results
        if view_img:
            cv2.imshow("YOLOv8-Face ONNX Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Save results
        if save_img:
            if source_type == "image":
                save_path = f"result_{os.path.basename(source)}"
                cv2.imwrite(save_path, frame)
                print(f"Result saved to {save_path}")
            else:
                vid_writer.write(frame)

        # Break after first frame for image
        if source_type == "image":
            break

    # Cleanup
    if source_type in ["video", "webcam"]:
        cap.release()
    if vid_writer is not None:
        vid_writer.release()
        print(f"Result saved to {save_path}")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8-Face ONNX Inference")
    parser.add_argument("--weights", type=str, default="weights/yolov8n-face.onnx", help="Path to ONNX model file")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video file or webcam index")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=750, help="Maximum detections per image")
    parser.add_argument("--save-img", action="store_true", help="Save detected images")
    parser.add_argument("--view-img", action="store_true", help="Display results (auto-enabled for webcam)")

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    run_face_detection(
        weights=args.weights,
        source=args.source,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        save_img=args.save_img,
        view_img=args.view_img,
    )


if __name__ == "__main__":
    main()
