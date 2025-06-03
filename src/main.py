import ffmpeg
import cv2
import os
import whisper

# Ensure that the CA certificates are set for requests
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/cert.pem'


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

def transcribe_audio(audio_path, model_size="small"):
    """
    Transcribe audio using the Whisper model.
    :param audio_path: Path to the audio file to transcribe.
    :param model_size: Size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
    :return: List of transcription segments with start, end, and text.
    """

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    print(result)
    return result["segments"]  # list of {start, end, text}

def extract_audio(video_path, output_wav_path):
    """
    Extract audio from a video file and save it as a WAV file.
    :param video_path: Path to the input video file.
    :param output_wav_path: Path where the output WAV file will be saved.
    """
    
    ffmpeg.input(video_path).output(output_wav_path, ac=1, ar='16000').overwrite_output().run()

video_path = "../clips/IMG_7817.MOV"
frames_out_dir = "frames"
audio_path = "audio.wav"

extract_frames(video_path, frames_out_dir, frame_rate=5)
print("done extracting frames")
extract_audio(video_path, audio_path)
segments = transcribe_audio(audio_path)

# Print some transcript chunks
for seg in segments[:5]:
    print(f"[{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}")