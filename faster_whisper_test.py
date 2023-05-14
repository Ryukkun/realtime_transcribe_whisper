import pyaudio
import wave
import asyncio
import webrtcvad
import pyaudio
import time
import numpy as np

from ctranslate2 import get_cuda_device_count
from typing import List, Union, Optional, Literal
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel



SAMPLING_RATE = 16000
CHUNK = 480
CHANNELS = 1



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
                 min_chunksize:int=300,
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
        self.last_packet_id = 0
        self.split_task = None
        #self.audio_save = AudioSave()


    async def from_file(self, audio) -> List[str]:
        return await self.whisper.async_transcribe(audio)


    async def from_packet(self, audio:Union[bytes,np.ndarray]):
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
            if not isinstance(audio_np, np.ndarray):
                audio_np = np.frombuffer(audio_byte, dtype=np.int16)
            if self.packet_size == None:
                self.packet_size = audio_np.size / SAMPLING_RATE * 1000
            audio_np = audio_np.astype(np.float32) / 32768.0
            self.packet_buffer.append({'data':audio_np, 'id':self.last_packet_id, 'time':time.perf_counter()})
            self.last_packet_id += 1
            if self.split_task == None:
                loop = asyncio.get_event_loop()
                self.split_task = loop.create_task(self._audio_split())


    async def _audio_split(self):
        _id = None
        while self.packet_buffer:
            packet = self.packet_buffer[-1]
            if packet['id'] == _id:
                break
            
            else:
                _id = packet['id']
                sleep_time = (self.split_timelimit / 1000) - (time.perf_counter() - packet['time'])
                #print(sleep_time)
                if 0 < sleep_time:
                    await asyncio.sleep(sleep_time)
                
        if self.packet_buffer:
            if (self.min_chunksize / self.packet_size) < len(self.packet_buffer):
                self.speech_buffer.append(np.concatenate([_['data'] for _ in self.packet_buffer]))
                self._transcribe_from_buffer()
                #print('split')
            self.packet_buffer.clear()

        self.split_task = None


    def _transcribe_from_buffer(self):
        if not self.speech_buffer:
            return
        
        exe = self.whisper.exe
        audio = self.speech_buffer.pop(0)
        loop = asyncio.get_event_loop()
        exe.submit(self.__transcribe_from_buffer, audio, loop)


    def __transcribe_from_buffer(self, audio:np.ndarray, loop:asyncio.BaseEventLoop):
        segments = self.whisper.segments(audio)
        black_list = ['ご視聴ありがとうございました','ご視聴ありがとうございます' , 'チャンネル登録','次の動画でお会いしましょう']
        
        for segment in segments:
            text = segment.text

            # 文字数の予測
            audio_size = audio.size / SAMPLING_RATE

            if len(text) < 50:
                if (audio_size*13) < len(text): 
                    #print(f'continue : {text} , 予想 : {audio_size*12}')
                    continue

                if (audio_size*8) < len(text):
                    #print('予測以上',end=' : ')
                    for _ in black_list:
                        if _ in text:
                            #print(f'continue : {text}')
                            continue
            
            asyncio.run_coroutine_threadsafe(self.callback(text), loop)



    async def callback(self, text:str):
        print(text)








if __name__ == "__main__":

    transcriber = RealtimeAudioTranscriber(whisper_module='small')

    print("使用可能なオーディオデバイス:")
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    default_host_api_index = audio.get_default_host_api_info()["index"]

    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        if (
            0 < device_info["maxInputChannels"] and
            device_info["hostApi"] == default_host_api_index
        ):
            print(f"Index: {device_info['index']}, Name: {device_info['name']}")


    # 対象のDeviceIndexを入力
    selected_device_index = int(input("対象のDeviceIndexを入力してください: "))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exe = ThreadPoolExecutor(1)
    exe.submit(loop.run_forever)



    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLING_RATE,
        input=True,
        input_device_index=selected_device_index,
    )


    print("Listening...")
    while True:
        audio = stream.read(CHUNK)
        asyncio.run_coroutine_threadsafe(transcriber.from_packet(audio), loop)