import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import supervision as sv


def process_video(model_path, video_path, output_dir):
    # Load YOLO model
    model = YOLO(model_path)

    # Create output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"❌ Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"helmet_detection_{timestamp}.mp4")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_count = 0
    print("⚡ Real-time helmet detection started! Press 'Q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO prediction
        results = model(frame, verbose=False)[0]

        # Convert results to supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Create readable labels
        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Write to file
        out.write(annotated_frame)

        #  Show live preview
        cv2.imshow("Helmet Detection (Press Q to quit)", annotated_frame)

        # Exit if Q pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n Stopped by user.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f" Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Processing complete! Output saved at: {output_path}")

# 🧩 NEW FUNCTION — for single image detection
def process_image(model_path, image_path, output_dir):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f" Could not load image file: {image_path}")

    # Run YOLO inference
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Labels for detected objects
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    # Save and show
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"helmet_detection_{timestamp}.jpg")
    cv2.imwrite(output_path, annotated_image)

    cv2.imshow("Helmet Detection - Image", annotated_image)
    print(f" Image processed successfully! Saved at: {output_path}")
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()