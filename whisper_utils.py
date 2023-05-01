from faster_whisper import WhisperModel


class WhisperModelWrapper:
    def __init__(self):
        self.model_size_or_path = "small"
        self.model = WhisperModel(
            self.model_size_or_path, device="cpu", compute_type="int8"
        )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio=audio, beam_size=3, language="ja", without_timestamps=True 
        )
        return segments