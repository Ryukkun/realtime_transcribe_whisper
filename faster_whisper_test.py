from audio_utils import display_valid_input_devices, create_audio_stream

import pyaudio
import wave
import asyncio
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
MODULE = 'small'
DEVICE = 'cuda'
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



class RealtimeAudioTranscriber:
    def __init__(self, packet_size_ms=20):
        self.whisper = WhisperModelWrapper()
        self.vad = webrtcvad.Vad(3)
        self.packet_size_ms = packet_size_ms
        self.speech_buffer = []
        self.packet_buffer = []
        self.audio_save = AudioSave()
        self.loop = asyncio.get_event_loop()
        self.packet_uid = 0


    async def from_file(self, audio) -> List[str]:
        return await self.whisper.async_transcribe(audio)


    def from_packet(self, audio:np.ndarray):
        """リアルタイム文字起こし

        Parameters
        ----------
        audio : np.ndarray
            サンプリングレートは 16000Hz 
            長さは 10ms, 20ms, 30ms
            int16
        """
        audio_byte = audio.tobytes()
        print(f'{len(self.speech_buffer)} : {len(self.packet_buffer)}')
        is_speech = self.vad.is_speech(audio_byte, SAMPLING_RATE)
        

        if is_speech:
            #print('True')
            audio = audio.astype(np.float32) / 32768.0
            self.packet_buffer.append(audio)
            self.packet_uid += 1
            self.loop.create_task(self._audio_split(self.packet_uid))
            


    async def _audio_split(self, res):
        await asyncio.sleep(SPLIT_TIME_LIMIT / 1000)
        if self.packet_buffer:
            if self.packet_uid == res:
                if (MIN_PACKET_TIME // self.packet_size_ms) < len(self.packet_buffer):
                    self.speech_buffer.append(np.concatenate(self.packet_buffer))
                    self.loop.create_task(self._transcribe_from_buffer())
                    print('split')
                self.packet_buffer.clear()


    async def _transcribe_from_buffer(self):
        if not self.speech_buffer:
            return
        
        exe = self.whisper.exe
        audio = self.speech_buffer.pop(0)
        await self.loop.run_in_executor(exe, self.__transcribe_from_buffer, audio)


    def __transcribe_from_buffer(self, audio:np.ndarray):
        segments = self.whisper.segments(audio)
        for segment in segments:
            text = segment.text

            # 文字数の予測
            text_len = audio.size // SAMPLING_RATE * 10
            if audio.size // SAMPLING_RATE * 10 < len(text):
                print(f'continue : {text} \n 予想文字数 : {text_len}')
                continue
            
            self.loop.create_task(self.callback(text))


    async def callback(self, text):
        print(text)








if __name__ == "__main__":

    async def main():
        transcriber = RealtimeAudioTranscriber()

        print("使用可能なオーディオデバイス:")
        display_valid_input_devices()

        # 対象のDeviceIndexを入力
        #selected_device_index = int(input("対象のDeviceIndexを入力してください: "))
        loop = asyncio.get_event_loop()

        def process_audio(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.int16)
            transcriber.from_packet(audio)
            print('kitya!',end='')
            return (in_data, pyaudio.paContinue)

        def process_loop(stream:pyaudio.Stream):
            while True:
                audio = np.frombuffer( stream.read(480), dtype=np.int16)
                transcriber.from_packet(audio)


        async def run():
            RATE = 16000
            CHUNK = 480
            FORMAT = pyaudio.paInt16
            CHANNELS = 1

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=2,
                frames_per_buffer=CHUNK,
            )

            print("Listening...")
            await loop.run_in_executor(None, process_loop, stream)




        # 文字起こしを開始
        await run()

        while True:
            await asyncio.sleep(10)

    asyncio.get_event_loop().run_until_complete(main())