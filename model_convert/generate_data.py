import sys
sys.path.append("..")
from utils_vad import *
import os
import onnxruntime as ort
import numpy as np
from tqdm import trange, tqdm
import tarfile


batch_size = 1
sr = 16000
hidden_size = 128
context_size = 64 if sr == 16000 else 32
context = np.zeros((1, context_size), dtype=np.float32)
state = np.zeros((2, 1, hidden_size), dtype=np.float32)
num_samples = 512 if sr == 16000 else 256
max_num = 200

calib_path = "calibration_dataset"
os.makedirs(calib_path, exist_ok=True)

input_names = ["data", "state"]
tars = {}
for name in input_names:
    os.makedirs(f"{calib_path}/{name}", exist_ok=True)
    
ort_model = ort.InferenceSession("../silero_vad.onnx", providers=["CPUExecutionProvider"])

with open('wavlist.txt', 'r') as f:
    for line in tqdm(f):
        line = line.strip()
        wav = read_audio(line)
        wav_name = os.path.splitext(os.path.basename(line))[0]
        

        if wav.shape[0] % num_samples:
            pad_num = num_samples - (wav.shape[0] % num_samples)
            wav = np.pad(wav, ((0, pad_num)), 'constant')

        index = 0
        for i in range(0, wav.shape[0], num_samples):
            wavs_batch = wav[i:i+num_samples][None, ...]
            data = np.concatenate([context, wavs_batch], axis=1)
            data = np.pad(data, ((0, 0), (0, 64)), 'reflect')
            input_feed = {
                "data": data,
                "state": state
            }

            for k in input_feed.keys():
                os.makedirs(f"{calib_path}/{k}/{wav_name}", exist_ok=True)
                np.save(f"{calib_path}/{k}/{wav_name}/{index}.npy", input_feed[k])
            
            output, state = ort_model.run(None, input_feed)
            output = np.array([output], dtype=np.float32)

            context = wavs_batch[..., -context_size:]
            index += 1
            if max_num is not None and index >= max_num:
                print(f"Exceed max_num {max_num}, break")
                break

for name in input_names:
    tars[name] = tarfile.open(f"{calib_path}/{name}.tar.gz", "w:gz")
    tars[name].add(f"{calib_path}/{name}")
    tars[name].close()