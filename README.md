# silero-vad.axera

Silero VAD implementation on Axera platforms

Thanks to https://github.com/lovemefan/Silero-vad-pytorch/tree/main, a reverse engineering implementation of https://github.com/snakers4/silero-vad

仓库自带两个芯片的预编译模型：

| 芯片 | NPU | 板子型号 | axmodel |
|------|-----|----------|---------|
| AX650 | NPU3 | AX650N | `src/silero_vad_axera/data/silero_vad_ax650.axmodel` |
| AX620E | NPU2 | AX630C | `src/silero_vad_axera/data/silero_vad_ax630c.axmodel` |

本仓库只支持 **axmodel（NPU）推理**，不提供 onnxruntime/CPU 后端；
ONNX 仅作为编译 axmodel 的中间产物（见「从零复现」）。

推理封装与后处理与官方 [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
（master）对齐：`model(chunk, sampling_rate)` 调用约定、`get_speech_timestamps`
（含 `time_resolution` / `use_max_poss_sil_at_max_speech` 等新参数）、
`VADIterator`、`collect_chunks` / `drop_chunks`（支持秒坐标）。
注意：编译出的 axmodel 是 16k 静态图，仅支持 `sampling_rate=16000`
（或其整数倍，会自动降采样），不支持 8000。

## 从零复现（x86 上导出 + 编译）

### 1. 准备环境并导出 ONNX

```bash
pip install -r model_convert/requirements.txt
```

下载官方 JIT 模型到 `model_convert/silero_vad.jit`：

```bash
git clone --depth 1 https://github.com/snakers4/silero-vad.git /tmp/silero-vad
cp /tmp/silero-vad/src/silero_vad/data/silero_vad.jit model_convert/
```

导出 `silero_vad.onnx`：

```bash
cd model_convert
python export_onnx.py
```

### 2. 生成校准数据

```bash
python generate_data.py
```

读取 `wavlist.txt` 中的 wav（默认 `../en.wav` 与 `../tests/data/test.wav`），
生成 `calibration_dataset/data.tar.gz` 与 `calibration_dataset/state.tar.gz`。

### 3. 编译 axmodel（需要 docker + Pulsar2 镜像，默认 `pulsar2:7.0`）

```bash
./compile_axmodel.sh
```

同时生成两个芯片的 axmodel，输出到 `src/silero_vad_axera/data/`。
本机已有 Pulsar2 可执行文件时：`PULSAR2_CMD=/opt/pulsar2/bin/pulsar2 ./compile_axmodel.sh`。

## 板端运行

### 安装

```bash
pip install -e .
```

`pip install -e .` 会自动安装 pyaxengine（axengine，来自官方 GitHub release：
`https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc2/axengine-0.1.3-py3-none-any.whl`）。
若板端无法直连 GitHub，可先在 x86 上下载该 wheel 后手动 `pip install axengine-0.1.3-py3-none-any.whl`。

其他人也可以直接从 GitHub 安装（无需发布到 PyPI）：

```bash
# 方式一：从仓库直接安装（推荐，装的就是 main 最新代码）
pip install "silero-vad-axera @ git+https://github.com/ml-inory/silero-vad.axera.git"

# 方式二：安装固定版本的 release wheel
pip install https://github.com/ml-inory/silero-vad.axera/releases/download/v0.1.1/silero_vad_axera-0.1.1-py3-none-any.whl
```

> 关于 PyPI：PyPI 上已存在旧版 `silero-vad-axera 0.1.1`（不含本仓库的修复）；
> 修复后的版本已发布到 PyPI（`silero-vad-axera 0.1.2`）：
> ```bash
> pip install silero-vad-axera==0.1.2
> ```
> 因 PyPI 禁止依赖中的直链 URL，而 `axengine` 只发布在 GitHub（不在 PyPI），
> PyPI 版不含 axengine 依赖，装完需按上文手动安装 axengine wheel；
> GitHub 安装方式则一条命令装全（推荐）。

> AX620E/AX630C 板端需已安装 NPU 运行库 `libax_engine.so`（官方固件自带；
> 若缺失，从对应 BSP SDK 把 `libax_engine.so` / `libax_sys.so` / `libax_interpreter.so`
> 放到 `/usr/local/lib`（或 `/soc/lib`）并执行 `ldconfig`）。

### 示例

```bash
python example.py --backend ax650    # AX650 板
python example.py --backend ax630c   # AX620E/AX630C 板
```

读取 `en.wav`，生成 `only_speech.wav`，`only_speech.wav` 仅包含 `en.wav` 中有说话的部分。

## 测试

```bash
pip install -e .[test]
pytest tests/
```

测试不依赖 NPU：验证 axmodel 数据完整性 + 后处理逻辑（用假模型模拟概率输出）。

## 上传到 PyPI

```bash
python -m build --sdist --wheel
python -m twine upload dist/*
```
