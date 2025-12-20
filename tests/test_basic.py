from silero_vad_axera import load_silero_vad, read_audio, get_speech_timestamps


def test_onnx_model():
    model = load_silero_vad(backend='onnx')
    for path in ["tests/data/test.wav",]:
        audio = read_audio(path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(audio, model, visualize_probs=False, return_seconds=True)
        assert speech_timestamps is not None

        out = model.audio_forward(audio, sr=16000)
        assert out is not None