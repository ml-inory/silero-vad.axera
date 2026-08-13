import numpy as np

from silero_vad_axera import (collect_chunks, drop_chunks, get_speech_timestamps,
                              VADIterator)


class FakeVAD:
    """无 NPU 环境下验证后处理逻辑的假模型，行为对齐 model(x, sr) 约定。"""

    def __init__(self, prob_fn):
        self._prob_fn = prob_fn
        self.reset_states()

    def reset_states(self, batch_size=1):
        self._i = 0

    def __call__(self, x, sr: int = 16000):
        p = self._prob_fn(self._i)
        self._i += 1
        return np.array([[p]], dtype=np.float32)

    def audio_forward(self, x, sr: int = 16000):
        self.reset_states()
        n = x.shape[1] if x.ndim == 2 else len(x)
        outs = []
        for i in range(0, n, 512):
            chunk = x[..., i:i+512] if x.ndim == 2 else x[i:i+512]
            outs.append(self.__call__(chunk, sr))
        return np.concatenate(outs, axis=-1)


def test_package_data_has_axmodels():
    from importlib import resources
    for name in ("silero_vad_ax650.axmodel", "silero_vad_ax630c.axmodel"):
        p = resources.files("silero_vad_axera.data").joinpath(name)
        assert p.is_file() and p.stat().st_size > 0


def test_get_speech_timestamps_all_speech():
    model = FakeVAD(lambda i: 0.9)
    audio = np.zeros(16000)
    st = get_speech_timestamps(audio, model, sampling_rate=16000, min_speech_duration_ms=100)
    assert len(st) == 1
    assert st[0]['start'] <= 0 and st[0]['end'] >= 16000


def test_get_speech_timestamps_no_speech():
    model = FakeVAD(lambda i: 0.01)
    audio = np.zeros(16000)
    st = get_speech_timestamps(audio, model, sampling_rate=16000)
    assert st == []


def test_vad_iterator():
    probs = [0.01, 0.01, 0.9, 0.9, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01]
    model = FakeVAD(lambda i: probs[i])
    it = VADIterator(model, sampling_rate=16000)
    events = []
    for _ in range(len(probs)):
        e = it(np.zeros(512), return_seconds=False)
        if e:
            events.append(e)
    assert events[0] == {'start': 544}        # 3*512 - 480 - 512
    assert events[1] == {'end': 3040}         # 6*512 + 480 - 512


def test_collect_and_drop_chunks_seconds():
    wav = np.arange(16000, dtype=np.float32)
    tss = [{'start': 0.1, 'end': 0.2}, {'start': 0.5, 'end': 0.6}]
    collected = collect_chunks(tss, wav, seconds=True, sampling_rate=16000)
    assert len(collected) == 3200
    dropped = drop_chunks(tss, wav, seconds=True, sampling_rate=16000)
    assert len(dropped) == 16000 - 3200
