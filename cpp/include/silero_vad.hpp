#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Silero VAD on AX NPU（AX650 NPU3 / AX620E NPU2），行为与 Python SDK 对齐：
//   data  [1, 640] float32 = context(64) + 512 样本 + reflect pad(64)
//   state [2, 1, 128] float32
//   -> output [1,1]（语音概率）/ next_state [2,1,128]
// 音频要求：16kHz 单声道；支持 16000 的整数倍采样率需先自行降采样到 16k。
class SileroVAD {
public:
    explicit SileroVAD(const std::string& model_path,
                       const std::string& model_name = "silero_vad");
    ~SileroVAD();

    SileroVAD(const SileroVAD&) = delete;
    SileroVAD& operator=(const SileroVAD&) = delete;

    // 清空 context / state（等价 Python model.reset_states()）
    void Reset();

    // 处理一帧：512 个 int16 样本（按 /32768 归一化，与 librosa 一致）
    float ProcessFrame(const int16_t* pcm512);

    // 处理一帧：512 个 float 样本（[-1, 1]）
    float ProcessFrame(const float* samples512);

    // 整段音频 forward（自动补零到 512 的整数倍），返回逐帧语音概率
    std::vector<float> AudioForward(const int16_t* pcm, size_t n);
    std::vector<float> AudioForward(const float* samples, size_t n);

    static constexpr int kSampleRate = 16000;
    static constexpr int kNumSamples = 512;
    static constexpr int kContextSize = 64;
    static constexpr int kStateSize = 2 * 128;

private:
    struct Impl;
    Impl* impl_;
};
