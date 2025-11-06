import os
from utils.helperFunctions import process_video, process_image

#  Absolute paths to your files
MODEL_PATH = "Helmet_Detection_Video_YOLOv8\Model\hemletYoloV8_100epochs.pt"
VIDEO_PATH = "Helmet_Detection_Video_YOLOv8\input/126434-735976920_small.mp4"
IMAGE_PATH = "Helmet_Detection_Video_YOLOv8\input\image.png"
OUTPUT_DIR = "Helmet_Detection_Video_YOLOv8/runs\outputs"

def main():
    print(" Helmet Detection started...\n")

    # ==== CHOOSE MODE ====
    MODE = "video"   # change to "video" or "image"

    if MODE == "video":
        print(f"🎥 Input Video: {VIDEO_PATH}")
        print(f"💾 Output will be saved at: {OUTPUT_DIR}\n")
        process_video(MODEL_PATH, VIDEO_PATH, OUTPUT_DIR)

    elif MODE == "image":
        print(f" Input Image: {IMAGE_PATH}")
        print(f" Output will be saved at: {OUTPUT_DIR}\n")
        process_image(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR)

    else:
        print("❌ Invalid MODE! Please choose either 'video' or 'image'.")

if __name__ == "__main__":
    main()
