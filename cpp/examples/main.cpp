// silero-vad.axera C++ 示例：读取 16k 单声道 s16le raw PCM，
// 输出逐帧语音概率与简单的语音区间（阈值 0.5 / 0.35，最短静音约 100ms）。
#include "silero_vad.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<int16_t> read_pcm_s16le(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(std::string("failed to open ") + path);
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (bytes.size() % 2) bytes.pop_back();
    std::vector<int16_t> pcm;
    pcm.reserve(bytes.size() / 2);
    for (size_t i = 0; i < bytes.size(); i += 2) {
        pcm.push_back(static_cast<int16_t>(bytes[i] | (bytes[i + 1] << 8)));
    }
    return pcm;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.axmodel> <pcm_s16le_16k.raw> [--probs]\n",
                     argv[0]);
        return 1;
    }
    const bool dump_probs = argc >= 4 && std::string(argv[3]) == "--probs";

    try {
        const std::vector<int16_t> pcm = read_pcm_s16le(argv[2]);
        if (pcm.size() < SileroVAD::kNumSamples) {
            throw std::runtime_error("PCM too short");
        }

        SileroVAD vad(argv[1]);
        const std::vector<float> probs = vad.AudioForward(pcm.data(), pcm.size());

        if (dump_probs) {
            for (float p : probs) std::printf("%.6f\n", p);
            return 0;
        }

        // 简单区间状态机：prob>=0.5 开始，prob<0.35 且持续 >=3 帧(≈100ms)结束
        constexpr float kThreshold = 0.5f;
        constexpr float kNegThreshold = 0.35f;
        constexpr int kMinSilenceFrames = 3;

        bool triggered = false;
        int temp_end = -1;
        int speech_frames = 0;
        std::vector<std::pair<int, int>> segments;  // {start_frame, end_frame}
        int seg_start = 0;

        for (size_t i = 0; i < probs.size(); ++i) {
            const float p = probs[i];
            if (p >= kThreshold) ++speech_frames;
            if (p >= kThreshold && temp_end >= 0) temp_end = -1;
            if (p >= kThreshold && !triggered) {
                triggered = true;
                seg_start = static_cast<int>(i);
            }
            if (triggered && p < kNegThreshold) {
                if (temp_end < 0) temp_end = static_cast<int>(i);
                if (static_cast<int>(i) - temp_end >= kMinSilenceFrames - 1) {
                    segments.emplace_back(seg_start, temp_end);
                    triggered = false;
                    temp_end = -1;
                }
            }
        }
        if (triggered) segments.emplace_back(seg_start, static_cast<int>(probs.size()) - 1);

        std::printf("frames=%zu speech_frames=%d segments=%zu\n",
                    probs.size(), speech_frames, segments.size());
        for (const auto& seg : segments) {
            std::printf("speech %.3fs - %.3fs (samples %d - %d)\n",
                        seg.first * 0.032, (seg.second + 1) * 0.032,
                        seg.first * SileroVAD::kNumSamples,
                        (seg.second + 1) * SileroVAD::kNumSamples);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
