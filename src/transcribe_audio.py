import whisper

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