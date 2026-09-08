#include "silero_vad.hpp"

#include "model_runner.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

constexpr int SileroVAD::kSampleRate;
constexpr int SileroVAD::kNumSamples;
constexpr int SileroVAD::kContextSize;
constexpr int SileroVAD::kStateSize;

namespace {

void reflect_pad_right(float* data, int n, int pad) {
    // numpy 'reflect'（Python SDK np.pad(..., 'reflect')）：不重复边缘元素
    // data[n-1+k] = data[n-1-k]，k = 1..pad
    for (int k = 1; k <= pad; ++k) {
        data[n - 1 + k] = data[n - 1 - k];
    }
}

}  // namespace

struct SileroVAD::Impl {
    ModelRunner runner;
    std::array<float, kContextSize> context{};
    std::array<float, kStateSize> state{};
    std::array<float, kNumSamples> frame{};

    explicit Impl(const std::string& model_path, const std::string& model_name)
        : runner(model_path, model_name) {}
};

SileroVAD::SileroVAD(const std::string& model_path, const std::string& model_name)
    : impl_(new Impl(model_path, model_name)) {}

SileroVAD::~SileroVAD() {
    delete impl_;
}

void SileroVAD::Reset() {
    impl_->context.fill(0.f);
    impl_->state.fill(0.f);
}

float SileroVAD::ProcessFrame(const int16_t* pcm512) {
    for (int i = 0; i < kNumSamples; ++i) {
        impl_->frame[i] = static_cast<float>(pcm512[i]) / 32768.f;
    }
    return ProcessFrame(impl_->frame.data());
}

float SileroVAD::ProcessFrame(const float* samples512) {
    std::array<float, kNumSamples + kContextSize + kContextSize> data{};
    float* raw = data.data() + kContextSize;  // 前 64 为 context

    std::memcpy(data.data(), impl_->context.data(), kContextSize * sizeof(float));
    std::memcpy(raw, samples512, kNumSamples * sizeof(float));
    reflect_pad_right(data.data(), kNumSamples + kContextSize, kContextSize);

    std::vector<float> in_data(data.begin(), data.end());
    std::vector<float> in_state(impl_->state.begin(), impl_->state.end());
    std::vector<std::vector<float>> outs = impl_->runner.Run({in_data, in_state});

    if (outs.size() < 2 || outs[0].empty() ||
        outs[1].size() * sizeof(float) < static_cast<size_t>(kStateSize)) {
        throw std::runtime_error("unexpected model outputs (expect output + next_state)");
    }

    const float prob = outs[0][0];
    std::memcpy(impl_->state.data(), outs[1].data(), kStateSize * sizeof(float));
    std::memcpy(impl_->context.data(), samples512 + kNumSamples - kContextSize,
                kContextSize * sizeof(float));
    return prob;
}

std::vector<float> SileroVAD::AudioForward(const int16_t* pcm, size_t n) {
    std::vector<float> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = static_cast<float>(pcm[i]) / 32768.f;
    }
    return AudioForward(samples.data(), samples.size());
}

std::vector<float> SileroVAD::AudioForward(const float* samples, size_t n) {
    std::vector<float> work(samples, samples + n);
    if (work.size() % kNumSamples) {
        work.resize(work.size() + (kNumSamples - work.size() % kNumSamples), 0.f);
    }

    Reset();
    std::vector<float> probs;
    probs.reserve(work.size() / kNumSamples);
    for (size_t i = 0; i < work.size(); i += kNumSamples) {
        probs.push_back(ProcessFrame(work.data() + i));
    }
    return probs;
}
