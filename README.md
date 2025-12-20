# silero-vad.axera
Silero VAD implementation on Axera platforms

Thanks to https://github.com/lovemefan/Silero-vad-pytorch/tree/main, a reverse engineering implementation of https://github.com/snakers4/silero-vad


## 导出ONNX
```
python export_onnx.py
```
生成silero_vad.onnx

## 对比ONNX和PyTorch
```
python compare.py
```

## 示例

### backend

如要使用ax后端，先安装pyaxengine:
```
pip install https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc2/axengine-0.1.3-py3-none-any.whl
```

```
python example.py
```
读取en.wav，生成only_speech.wav，only_speech.wav仅包含en.wav中有说话的部分