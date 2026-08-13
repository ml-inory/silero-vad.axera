import numpy as np
import axengine as axe


class SileroAx:
    """AX NPU 推理封装，调用约定对齐官方 silero-vad OnnxWrapper：__call__(x, sr)。

    说明：编译出的 axmodel 为 16k 静态图（data=1x640），因此仅支持
    sr=16000（或其整数倍，会自动降采样），不支持 8000。
    """

    def __init__(self, path: str, providers=['AxEngineExecutionProvider']):
        super().__init__()

        self.batch_size = 1
        self.sample_rates = [16000]
        self._state = np.zeros((2, self.batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

        self.model = axe.InferenceSession(path, providers=providers)

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x, sr: int):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, ...]
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def __call__(self, x, sr: int = 16000):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported values: 512 for 16000 sample rate)"
            )

        batch_size = x.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"当前 axmodel 仅支持 batch_size=1，收到 {batch_size}")
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        data = np.concatenate([self._context, x], axis=1)
        data = np.pad(data, ((0, 0), (0, 64)), 'reflect')
        input_feed = {
            "data": data,
            "state": self._state
        }

        output, self._state = self.model.run(None, input_feed=input_feed)
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        if len(output.shape) == 0:
            output = np.array([output], dtype=np.float32)

        return output

    def audio_forward(self, x, sr: int):
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), 'constant', value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        return np.concatenate(outs, axis=-1)
