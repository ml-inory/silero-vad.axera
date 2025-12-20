import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="ax650", choices=['ax650', 'ax630c', 'onnx'])
args = parser.parse_args()

from silero_vad_axera import *
from pprint import pprint

SAMPLING_RATE = 16000
model = load_silero_vad(args.backend)
wav_path = "en.wav"

""" Speech timestamps from full audio """
wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)

# merge all speech chunks to one audio
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 


""" Entire audio inference """
wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
# chunk size is 32 ms, and each second of the audio contains 31.25 chunks
# currently only chunks of size 512 are used for 16 kHz and 256 for 8 kHz
# e.g. 512 / 16000 = 256 / 8000 = 0.032 s = 32.0 ms
predicts = model.audio_forward(wav, sr=SAMPLING_RATE)


""" Stream imitation example """
## using VADIterator class

vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)
wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)

window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio

## just probabilities

wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+window_size_samples]
    if len(chunk) < window_size_samples:
        break
    speech_prob = model(chunk).item()
    speech_probs.append(speech_prob)
model.reset_states() # reset model states after each audio

print(speech_probs[:10]) # first 10 chunks predicts