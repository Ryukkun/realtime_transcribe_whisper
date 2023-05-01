import soundcard as sc
import numpy as np

SAMPLE_RATE = 16000
INTERVAL = 3
BUFFER_SIZE = 4096

# start recording
with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
    n = 0
    while True:
        while n < SAMPLE_RATE * INTERVAL:
            data = mic.record(BUFFER_SIZE)
            audio[n:n+len(data)] = data.reshape(-1)
            n += len(data)

        # find silent periods
        m = n * 4 // 5
        vol = np.convolve(audio[m:n] ** 2, b, 'same')
        m += vol.argmin()
        q.put(audio[:m])

        audio_prev = audio
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        audio[:n-m] = audio_prev[m:n]
        n = n-m