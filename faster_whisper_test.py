from audio_utils import display_valid_input_devices, create_audio_stream

import wave
import asyncio
import queue
import webrtcvad
import pyaudio
import numpy as np

from faster_whisper import WhisperModel



SAMPLING_RATE = 16000
SILENT_CHUNKS = 10
MIN_CHUNKS = 5


class AudioSave:
    def __init__(self) -> None:
        self.i = 0
        self.ac = 1
        self.abit = 2

    def save(self, data):
        with wave.Wave_write(f'{self.i}.wav') as f:
            f.setsampwidth(self.abit)
            f.setnchannels(self.ac)
            f.setframerate(SAMPLING_RATE)
            f.writeframes(data)
        self.i += 1

class WhisperModelWrapper:
    def __init__(self):
        self.model_size_or_path = "tiny"
        self.model = WhisperModel(
            self.model_size_or_path, device="cpu", compute_type="float32"
        )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio=audio, beam_size=2, language="ja", without_timestamps=True 
        )
        return segments


class AudioTranscriber:
    def __init__(self):
        self.model_wrapper = WhisperModelWrapper()
        self.vad_wrapper = webrtcvad.Vad(0)
        self.silent_chunks = 0
        self.speech_buffer = []
        self.audio_queue = queue.Queue()
        self.audio_save = AudioSave()

    async def transcribe_audio(self):
        loop = asyncio.get_event_loop()
        while True:
            while self.audio_queue.empty():
                await asyncio.sleep(0.1)
            audio_data_np = self.audio_queue.get()
            segments = await loop.run_in_executor(None, self.model_wrapper.transcribe, audio_data_np)

            for segment in segments:
                text = segment.text
                if not 'ご視聴ありがとうございました' in text:
                    print(segment.text)
            #self.audio_save.save(audio_data_np)
        
        await loop.run_in_executor(None, self.model_wrapper.transcribe, audio_data_np)
        

    def process_audio(self, in_data, frame_count, time_info, status):
        is_speech = self.vad_wrapper.is_speech(in_data, SAMPLING_RATE)
        #rint(f'{len(self.speech_buffer)} : {self.audio_queue.qsize()}')

        if is_speech:
            #print('True')
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            self.speech_buffer.append(audio_data)
            self.silent_chunks = 0
        else:
            self.silent_chunks += 1


        if not is_speech and SILENT_CHUNKS < self.silent_chunks:
            if MIN_CHUNKS < len(self.speech_buffer):
                audio_data_np = np.concatenate(self.speech_buffer)
                self.speech_buffer.clear()
                self.audio_queue.put(audio_data_np)
            else:
                # noise clear
                self.speech_buffer.clear()

        return (in_data, pyaudio.paContinue)

    async def start_transcription(self, selected_device_index):
        stream = create_audio_stream(selected_device_index, self.process_audio)
        print("Listening...")
        stream.start_stream()
        loop = asyncio.get_event_loop()
        return loop.create_task(self.transcribe_audio())






if __name__ == "__main__":

    async def main():
        transcriber = AudioTranscriber()

        print("使用可能なオーディオデバイス:")
        display_valid_input_devices()

        # 対象のDeviceIndexを入力
        selected_device_index = int(input("対象のDeviceIndexを入力してください: "))

        # 文字起こしを開始
        await transcriber.start_transcription(selected_device_index)

        while True:
            await asyncio.sleep(10)

    asyncio.run(main())