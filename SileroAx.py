import numpy as np
import axengine as axe


class SileroAx:
    def __init__(self, path: str, providers=['AxEngineExecutionProvider']):
        super().__init__()

        self.batch_size = 1
        self.sr = 16000
        self.hidden_size = 128
        self.context_size = 64 if self.sr == 16000 else 32
        self.context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        self.state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)
        self.num_samples = 512 if self.sr == 16000 else 256

        self.model = axe.InferenceSession(path, providers=providers)

    def reset_states(self):
        self.context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        self.state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)

    def __call__(self, x):
        if len(x.shape) == 1:
            x = x[None, ...]

        data = np.concatenate([self.context, x], axis=1)
        data = np.pad(data, ((0, 0), (0, 64)), 'reflect')
        input_feed = {
            "data": data,
            "state": self.state
        }

        output, self.state = self.model.run(None, input_feed=input_feed)
        self.context = x[..., -self.context_size:]

        if len(output.shape) == 0:
            output = np.array([output], dtype=np.float32)

        return output
    
    def audio_forward(self, x, sr):
        if len(x.shape) > 1:
            x = x[0]

        outs = []
        self.reset_states()
        num_samples = self.num_samples

        if x.shape[0] % num_samples:
            pad_num = num_samples - (x.shape[0] % num_samples)
            x = np.pad(x, ((0, pad_num)), 'constant', value=0.0)

        for i in range(0, x.shape[0], num_samples):
            wavs_batch = x[i:i+num_samples]
            out_chunk = self.__call__(wavs_batch)
            outs.append(out_chunk)

        stacked = np.concatenate(outs, axis=-1)
        return stacked