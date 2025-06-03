import ffmpeg

def extract_audio(video_path, output_wav_path):
    """
    Extract audio from a video file and save it as a WAV file.
    :param video_path: Path to the input video file.
    :param output_wav_path: Path where the output WAV file will be saved.
    """
    
    ffmpeg.input(video_path).output(output_wav_path, ac=1, ar='16000').overwrite_output().run()