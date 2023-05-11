from audio_utils import display_valid_input_devices, create_audio_stream

import wave
import asyncio
import queue
import webrtcvad
import pyaudio
import numpy as np

from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel



SAMPLING_RATE = 16000
SPLIT_TIME_LIMIT = 100 #ms
MIN_PACKET_TIME = 100

# Whisper
MODULE = 'tiny'
DEVICE = 'cpu'
TYPE = 'float32'


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
    exe = ThreadPoolExecutor(1)
    model = WhisperModel(
        MODULE, device=DEVICE, compute_type=TYPE
    )

    def segments(self, audio):
        segments, _ = self.model.transcribe(
            audio=audio, beam_size=2, language="ja", without_timestamps=True 
        )
        return segments


    def transcribe(self, audio):
        return list(self.segments(audio))
    

    async def async_transcribe(self, audio):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.exe, self.transcribe, audio)



class AudioTranscriber:
    def __init__(self, packet_size_ms=20):
        self.whisper = WhisperModelWrapper()
        self.vad = webrtcvad.Vad(0)
        self.packet_size = packet_size_ms
        self.packet_size_ms = packet_size_ms
        self.speech_buffer = []
        self.packet_buffer = []
        self.audio_save = AudioSave()


    async def from_file(self, audio) -> List[str]:
        return await self.whisper.async_transcribe(audio)


    async def from_packet(self, audio:np.ndarray):
        """リアルタイム文字起こし

        Parameters
        ----------
        audio : np.ndarray
            サンプリングレートは 16000Hz 
            長さは 10ms, 20ms, 30ms
            int16
        """
        audio_byte = audio.tobytes()
        is_speech = self.vad.is_speech(audio_byte, SAMPLING_RATE)
        #rint(f'{len(self.speech_buffer)} : {self.audio_queue.qsize()}')

        if is_speech:
            #print('True')
            audio = audio.astype(np.float32) / 32768.0
            self.packet_buffer.append(audio)
            loop = asyncio.get_event_loop()
            loop.create_task(self._audio_split(audio))
            


    async def _audio_split(self, audio:np.ndarray):
        await asyncio.sleep(SPLIT_TIME_LIMIT / 1000)
        if self.packet_buffer:
            if np.all(self.packet_buffer[-1] == audio):
                if (MIN_PACKET_TIME // self.packet_size_ms) < len(self.packet_buffer):
                    self.speech_buffer.append(np.concatenate(self.speech_buffer))
                self.speech_buffer.clear()


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