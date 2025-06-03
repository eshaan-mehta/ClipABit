import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video file and save them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param frame_rate: Number of frames to skip (1 means every frame, 2 means every second frame, etc.).
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate) # Number of frames to skip based on the frame rate

    frame_count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path} to {output_folder}.")