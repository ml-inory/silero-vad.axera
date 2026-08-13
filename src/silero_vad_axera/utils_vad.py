# -*- coding: utf-8 -*-
"""VAD 后处理工具，与 snakers4/silero-vad（master）的 utils_vad.py 对齐。

本文件是官方实现（torch/torchaudio）的 numpy 移植版：
- get_speech_timestamps / VADIterator / collect_chunks / drop_chunks 的参数、
  默认值与核心逻辑逐行对齐官方 master（含 time_resolution、
  use_max_poss_sil_at_max_speech、possible_ends 等新逻辑）；
- 读/写音频保留轻量依赖（librosa + soundfile），不引入 torch/torchaudio。

模型调用约定与原版一致：``model(chunk, sampling_rate)`` 返回概率（1x1）。
"""
import warnings
from typing import Callable, List

import librosa
import numpy as np
import soundfile as sf

languages = ['ru', 'en', 'de', 'es']


def read_audio(path: str,
               sampling_rate: int = 16000):
    wav, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return wav


def save_audio(path: str,
               tensor,
               sampling_rate: int = 16000):
    tensor = np.asarray(tensor)
    if tensor.ndim > 1:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor.mean(axis=0)
    sf.write(path, tensor, sampling_rate)


def make_visualization(probs, step):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('visualize_probs=True 需要 matplotlib，已跳过绘图')
        return
    plt.plot([x * step for x in range(len(probs))], probs)
    plt.ylim([0, 1.05])
    plt.xlim([0, len(probs) * step])
    plt.xlabel('seconds')
    plt.ylabel('speech probability')
    plt.show()


def get_speech_timestamps(audio,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          max_speech_duration_s: float = float('inf'),
                          min_silence_duration_ms: int = 100,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False,
                          time_resolution: int = 1,
                          visualize_probs: bool = False,
                          progress_tracking_callback: Callable[[float], None] = None,
                          neg_threshold: float = None,
                          window_size_samples: int = 512,
                          min_silence_at_max_speech: int = 98,
                          use_max_poss_sil_at_max_speech: bool = True):

    """
    与官方 silero-vad 一致：将长音频按 32ms(16k) 分块推理，
    通过状态机切分语音区间。返回值：list[dict]，'start'/'end' 为样本点
    （return_seconds=True 时为秒）。
    """
    if not isinstance(audio, np.ndarray):
        try:
            audio = np.asarray(audio, dtype=np.float32)
        except Exception:
            raise TypeError("Audio cannot be casted to numpy array. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                break
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError("Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates")

    window_size_samples = 512 if sampling_rate == 16000 else 256

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
        # calculate progress and send it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        # If speech returns after a temp_end, record candidate silence if long enough and clear temp_end
        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        # Start of speech
        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        # Max speech length reached: decide where to cut
        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])  # use the longest possible silence segment in the current speech chunk
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                # Legacy max-speech cut (use_max_poss_sil_at_max_speech=False): prefer last valid silence (prev_end) if available
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    # No prev_end -> fallback to cutting at current sample
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        # Silence detection while in speech
        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
            speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """流式 VAD：逐 chunk 调用，返回 {'start': ...} / {'end': ...} / None。"""

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """x: 音频 chunk（1D 或 2D (1, n)）"""

        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=np.float32)
            except Exception:
                raise TypeError("Audio cannot be casted to numpy array. Cast it manually")

        window_size_samples = len(x[0]) if x.ndim == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None


def collect_chunks(tss: List[dict],
                   wav: np.ndarray,
                   seconds: bool = False,
                   sampling_rate: int = None) -> np.ndarray:
    """按坐标列表从长音频中拼接语音片段（坐标可为样本点或秒）。"""
    if seconds and not sampling_rate:
        raise ValueError('sampling_rate must be provided when seconds is True')

    chunks = list()
    _tss = _seconds_to_samples_tss(tss, sampling_rate) if seconds else tss

    for i in _tss:
        chunks.append(wav[i['start']:i['end']])

    return np.concatenate(chunks)


def drop_chunks(tss: List[dict],
                wav: np.ndarray,
                seconds: bool = False,
                sampling_rate: int = None) -> np.ndarray:
    """按坐标列表从长音频中删除语音片段（坐标可为样本点或秒）。"""
    if seconds and not sampling_rate:
        raise ValueError('sampling_rate must be provided when seconds is True')

    chunks = list()
    cur_start = 0

    _tss = _seconds_to_samples_tss(tss, sampling_rate) if seconds else tss

    for i in _tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']

    chunks.append(wav[cur_start:])

    return np.concatenate(chunks)


def _seconds_to_samples_tss(tss: List[dict], sampling_rate: int) -> List[dict]:
    """把秒坐标转成样本坐标。"""
    return [{
        'start': round(crd['start'] * sampling_rate),
        'end': round(crd['end'] * sampling_rate)
    } for crd in tss]
