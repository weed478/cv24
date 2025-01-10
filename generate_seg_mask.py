import cv2
import numpy as np
from ultralytics import YOLO, SAM
import argparse

def process_video(video_path):
    # Initialize YOLO detection model and SAM model
    detect_model = YOLO('yolo11x.pt')
    sam_model = SAM('sam2.1_l.pt')

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object
    output_path = video_path.rsplit('.', 1)[0] + '_mask.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # First detect person with YOLO
        detections = detect_model(frame, classes=[0])  # class 0 is person

        # Create empty mask for this frame
        frame_mask = np.zeros((height, width), dtype=np.uint8)

        # Process first person detection if any
        if len(detections) > 0 and len(detections[0].boxes) > 0:
            # Get first person bounding box
            box = detections[0].boxes[0]  # get first detection
            bbox = box.xyxy[0].cpu().numpy()  # get bbox in xyxy format

            # Use SAM to generate mask from bbox
            results = sam_model(
                frame,
                bboxes=[bbox],  # pass the bounding box
                verbose=False
            )

            if results and results[0].masks is not None:
                # Get the mask from SAM
                mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

                # Resize mask to frame size
                frame_mask = cv2.resize(mask, (width, height))

        # Write the mask frame
        out.write(frame_mask)

        # Print progress
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')

        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', frame_mask)
        # cv2.waitKey(1)

    # Clean up
    cap.release()
    out.release()
    print("\nProcessing complete! Mask video saved as:", output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate segmentation mask video for persons')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    args = parser.parse_args()

    process_video(args.video_path)

if __name__ == "__main__":
    main()
