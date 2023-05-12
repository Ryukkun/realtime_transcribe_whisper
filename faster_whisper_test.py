from audio_utils import display_valid_input_devices, create_audio_stream

from ctranslate2 import get_cuda_device_count
import pyaudio
import wave
import asyncio
import webrtcvad
import pyaudio
import numpy as np

from typing import List, Union, Optional, Literal
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel



SAMPLING_RATE = 16000


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
    def __init__(self, model:Optional[str]=None, device:Optional[str]=None, _type:Optional[str]=None) -> None:
        """
        Whisper Module

        Parameters
        ----------
        model : Optional[str]
            model
        device : Optional[str]
            device
        _type : Optional[str]
            type
        """
        self.exe = ThreadPoolExecutor(1)
        self.model = WhisperModel(
            model, device=device, compute_type=_type
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


class Vad(webrtcvad.Vad):
    def __init__(self, mode=None):
        self.mode:int
        super().__init__(mode)
        

    def set_mode(self, mode):
        super().set_mode(mode)
        self.mode = mode


class RealtimeAudioTranscriber:
    def __init__(self, 
                 vad_mode:int=1, 
                 split_timelimit:int=100, 
                 min_chunksize:int=200,
                 whisper_device:Literal['cpu', 'cuda', 'auto'] = 'auto',
                 whisper_module:str = 'tiny',
                 whisper_type:str = 'float32',
                 whisper:Optional[WhisperModelWrapper] = None
                 ):
        """
        リアルタイムで文字起こしするから任せて
        音声を切り分けて送ってね！ 切り分けられた音声の長さは 10ms, 20ms, 30ms まで対応してるお！

        Parameters
        ----------
        vad_mode : int = 20
            喋っているか判別する時の感度 , by default 1
        split_timelimit : int = 100
            設定された時間だけvadで無音と判定されればwhisperに送られる(ms), default 100
        min_chunksize : int = 200
            vad・whisperの誤判定の防止。設定された時間だけvadで喋っていると見なされなければ、whisperで文字起こしを行わない(ms), by default 200
        whisper_device : Literal['cpu', 'cuda', 'auto'] = 'auto'
            cpu or cuda
        whisper_module : str = 'tiny'
            faster_whisper module
        whisper_type : str = 'float32'
            faster_whisper compute_type
        """
        if whisper:
            self.whisper = whisper
        else:
            if whisper_device == 'auto':
                if 0 == get_cuda_device_count():
                    whisper_device = 'cpu'
                else:
                    whisper_device = 'cuda'
            self.whisper = WhisperModelWrapper(model=whisper_module, device=whisper_device, _type=whisper_type)

        self.vad = Vad(vad_mode)
        self.split_timelimit = split_timelimit
        self.min_chunksize = min_chunksize
        self.packet_size:Optional[int] = None
        self.speech_buffer = []
        self.packet_buffer = []
        self.loop = asyncio.get_event_loop()
        self.last_packet_id = 0
        #self.audio_save = AudioSave()


    async def from_file(self, audio) -> List[str]:
        return await self.whisper.async_transcribe(audio)


    def from_packet(self, audio:Union[bytes,np.ndarray]):
        """
        サンプリングレートは 16000Hz
        ビット数は int16にしてくれよん

        Parameters
        ----------
        audio : Union[bytes,np.ndarray]
            オーディオパケットぷりーず
        """
        audio_byte: Optional[bytes] = None
        audio_np: Optional[np.ndarray] = None
        if isinstance(audio, bytes):
            audio_byte = audio

        else:
            audio_np = audio
            audio_byte = audio_np.tobytes()
        
        #print(f'{len(self.speech_buffer)} : {len(self.packet_buffer)}')
        is_speech = self.vad.is_speech(audio_byte, SAMPLING_RATE)
        
        if is_speech:
            print('True')
            if not isinstance(audio_np, np.ndarray):
                audio_np = np.frombuffer(audio_byte, dtype=np.int16)
            if self.packet_size == None:
                self.packet_size = audio_np.size / SAMPLING_RATE
            audio_np = audio_np.astype(np.float32) / 32768.0
            self.packet_buffer.append(audio_np)
            self.last_packet_id += 1
            asyncio.create_task(self._audio_split(self.last_packet_id))
            


    async def _audio_split(self, id):
        await asyncio.sleep(self.split_timelimit / 1000)
        if self.packet_buffer:
            if self.last_packet_id == id:
                if (self.min_chunksize / self.packet_size) < len(self.packet_buffer):
                    self.speech_buffer.append(np.concatenate(self.packet_buffer))
                    asyncio.create_task(self._transcribe_from_buffer())
                    print('split')
                self.packet_buffer.clear()


    async def _transcribe_from_buffer(self):
        if not self.speech_buffer:
            return
        
        exe = self.whisper.exe
        audio = self.speech_buffer.pop(0)
        loop = asyncio.get_event_loop()
        await asyncio.run_in_executor(exe, self.__transcribe_from_buffer, audio, loop)


    def __transcribe_from_buffer(self, audio:np.ndarray, loop:asyncio.BaseEventLoop):
        segments = self.whisper.segments(audio)
        for segment in segments:
            text = segment.text

            # 文字数の予測
            text_len = audio.size / SAMPLING_RATE * 12
            if text_len < len(text) < 30:
                print(f'continue : {text} \n 予想文字数 : {text_len}')
                continue
            
            loop.create_task(self.callback(text))


    async def callback(self, text):
        print(text)








if __name__ == "__main__":

    def main():
        transcriber = RealtimeAudioTranscriber()

        print("使用可能なオーディオデバイス:")
        display_valid_input_devices()

        # 対象のDeviceIndexを入力
        selected_device_index = int(input("対象のDeviceIndexを入力してください: "))
        loop = asyncio.get_event_loop()


        def process_loop(stream:pyaudio.Stream):
            while True:
                audio = stream.read( 480)
                audio = np.frombuffer( audio, dtype=np.int16)
                transcriber.from_packet(audio)


        def run():
            CHUNK = 480
            FORMAT = pyaudio.paInt16
            CHANNELS = 1

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLING_RATE,
                input=True,
                input_device_index=selected_device_index,
                frames_per_buffer=CHUNK,
            )

            print("Listening...")
            exe = ThreadPoolExecutor(1)
            task = exe.submit(process_loop, stream)
            task.result()




        # 文字起こしを開始
        run()


    main()
    #asyncio.get_event_loop().run_until_complete(main())