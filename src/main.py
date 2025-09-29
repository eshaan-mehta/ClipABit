import ffmpeg
import cv2
import os
import whisper
import subprocess
import json
import ssl
import certifi
import torch
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image

# Fix SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()

# Ensure that the CA certificates are set for requests
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/cert.pem'

def get_video_rotation(video_path):
    """
    Get the rotation metadata from a video file using ffprobe.
    Returns the rotation angle in degrees.
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    # Check for rotation in video streams
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            # Check for rotation in tags
            tags = stream.get('tags', {})
            if 'rotate' in tags:
                return int(tags['rotate'])
            # Check for rotation in side data
            for side_data in stream.get('side_data_list', []):
                if side_data.get('rotation'):
                    return side_data['rotation']
    return 0

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video file and save them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param frame_rate: Number of frames to skip (1 means every frame, 2 means every second frame, etc.).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get video rotation
    rotation = get_video_rotation(video_path)
    print(f"Detected video rotation: {rotation} degrees")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)  # Number of frames to skip based on the frame rate

    frame_count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            # Apply rotation if needed
            if rotation != 0:
                # Convert rotation to OpenCV rotation code
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            frame_filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
            print(f"Extracted frame {saved} from {video_path}")


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
    return result["segments"]  # list of {start, end, text}

def extract_audio(video_path, output_wav_path):
    """
    Extract audio from a video file and save it as a WAV file.
    :param video_path: Path to the input video file.
    :param output_wav_path: Path where the output WAV file will be saved.
    """
    
    ffmpeg.input(video_path).output(output_wav_path, ac=1, ar='16000').overwrite_output().run()

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def embed_images(frame_dir, output_path, interval_seconds=5):
    """
    Generate CLIP embeddings for extracted frames.
    :param frame_dir: Directory containing frames (jpg).
    :param output_path: JSONL file to write image embeddings.
    :param interval_seconds: Seconds between frames (used for timestamp).
    """
    model, preprocess, device = load_clip_model()
    embeddings = []

    for fname in sorted(os.listdir(frame_dir)):
        if not fname.endswith(".jpg"):
            continue
        path = os.path.join(frame_dir, fname)
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image).squeeze().cpu().tolist()
        # assume fname like frame_0000.jpg => timestamp = idx * interval_seconds
        idx = int(fname.split("_")[1].split(".")[0])
        embeddings.append({
            "type": "image",
            "frame_path": path,
            "timestamp": idx * interval_seconds,
            "embedding": feat
        })

    with open(output_path, "w") as f:
        for e in embeddings:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(embeddings)} image embeddings to {output_path}")

def embed_transcripts(segments, output_path):
    """
    Generate text embeddings for transcription segments.
    :param segments: List of dicts with 'start', 'end', 'text'.
    :param output_path: JSONL file to append text embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []

    for seg in segments:
        txt = seg["text"]
        emb = model.encode(txt).tolist()
        embeddings.append({
            "type": "text",
            "text": txt,
            "start": seg["start"],
            "end": seg["end"],
            "embedding": emb
        })

    with open(output_path, "a") as f:
        for e in embeddings:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(embeddings)} text embeddings to {output_path}")




video_path = "../clips/IMG_6874.MOV"
frames_out_dir = "frames"
audio_path = "audio.wav"

extract_frames(video_path, frames_out_dir, frame_rate=5)
print("done extracting frames")
extract_audio(video_path, audio_path)
segments = transcribe_audio(audio_path)
print("done transcribing audio")

embed_images(frames_out_dir, "embeddings.jsonl")
embed_transcripts(segments, "embeddings.jsonl")
print("done embedding")

# Print some transcript chunks
# for seg in segments[:5]:
#     print(f"[{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}")

