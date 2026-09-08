# silero_vad_axera C++ SDK

直接链接 AX Engine runtime（`ax_engine`/`ax_sys`/`ax_interpreter`）的
Silero VAD 流式推理 SDK，行为与 Python SDK 对齐：

- 输入（与 `model_meta` 一致）：`data[1,640]` float32（context 64 + 512 样本 + reflect pad 64）、
  `state[2,1,128]` float32
- 输出（与 `model_meta` 一致）：`output[1,1]` 语音概率、`next_state[2,1,128]`
- 支持芯片：AX650（NPU3，`silero_vad_ax650.axmodel`）、AX620E（NPU2，
  `silero_vad_ax630c.axmodel`，板子型号 AX630C）
- 音频：16kHz 单声道；int16 输入按 `/32768` 归一化（与 librosa 一致）

## 交叉编译（aarch64）

依赖：`aarch64-none-linux-gnu-gcc/g++`（GCC 9.2）与 AX runtime 头文件/库。
本目录 `axrt/` 已预置 AX650C runtime（include + libax_engine/libax_sys/
libax_interpreter），可直接编译；也可用你自己的 BSP 覆盖：

```bash
cmake -S cpp -B cpp/build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cpp/toolchain-aarch64.cmake \
  -DTOOLCHAIN_ROOT=/path/to/aarch64-none-linux-gnu \
  -DAX_RUNTIME_ROOT=/path/to/axrt        # 默认使用 cpp/axrt
cmake --build cpp/build-aarch64 -j
```

产物：`cpp/build-aarch64/silero_vad_example`。

## 板端运行

先把音频转成 16k 单声道 s16le raw PCM（例如用 ffmpeg / librosa），然后：

```bash
export LD_LIBRARY_PATH=/soc/lib      # 或指向板端 runtime 库目录
./silero_vad_example models/ax650/model.axmodel audio.raw          # AX650
./silero_vad_example models/ax630c/model.axmodel audio.raw         # AX620E/AX630C
./silero_vad_example model.axmodel audio.raw --probs               # 打印逐帧概率
```

模型文件在 `src/silero_vad_axera/data/`（`silero_vad_ax650.axmodel` /
`silero_vad_ax630c.axmodel`），也发布在 HF
（AXERA-TECH/SileroVAD `models/`）。

## 在自己的代码里用

```cpp
#include "silero_vad.hpp"

SileroVAD vad("silero_vad_ax650.axmodel", "silero_vad");
float prob = vad.ProcessFrame(pcm_512_int16);   // 每帧 512 个 int16 @16k
vad.Reset();                                    // 切换音频时重置状态
```

`SileroVAD` 内部自持 context(64) 与 state(2×128)，逐帧喂入即可；
`AudioForward()` 可一次处理整段音频并返回逐帧概率。
