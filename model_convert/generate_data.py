import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from silero_vad_axera.utils_vad import read_audio
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
    
ort_model = ort.InferenceSession(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx"),
    providers=["CPUExecutionProvider"],
)

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wavlist.txt'), 'r') as f:
    for line in tqdm(f):
        line = line.strip()
        wav_path = line if os.path.isabs(line) else os.path.join(repo_root, line)
        wav = read_audio(wav_path)
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
