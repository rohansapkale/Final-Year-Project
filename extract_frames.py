import cv2
import os

# Set how many frames you want per video
FRAMES_PER_VIDEO = 10

def extract_frames_from_video(video_path, output_folder, class_label, count_start=0):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = count_start

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // FRAMES_PER_VIDEO)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            filename = f"{class_label}_{saved_count}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

def process_videos(input_dir, output_dir, label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"[Warning] Input directory not found: {input_dir}")
        return

    for video_file in os.listdir(input_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(input_dir, video_file)
            extract_frames_from_video(video_path, output_dir, label)

# === MAIN EXECUTION ===
splits = ['train', 'valid', 'test']
base_input = r"dataset/real life violence situations"  # Raw string for spaces

base_output = 'dataset/frames'
for split in splits:
    for category in ['Violence', 'NonViolence']:
        input_path = os.path.join(base_input, split, category)
        output_path = os.path.join(base_output, split, category)
        label = category.lower()
        print(f"Looking for folder: {os.path.abspath(input_path)}")  # <== Debug print
        process_videos(input_path, output_path, label)
