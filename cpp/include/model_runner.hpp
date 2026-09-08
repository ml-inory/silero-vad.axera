#pragma once

#include <cstddef>
#include <string>
#include <vector>


class ModelRunner {
public:
    explicit ModelRunner(const std::string& model_path, const std::string& model_name = "model");
    ~ModelRunner();

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    size_t NumInputs() const;
    size_t NumOutputs() const;
    size_t InputBytes(size_t index) const;
    size_t OutputBytes(size_t index) const;

    // inputs[i] 为第 i 个输入的 float 数据；返回每个输出的 float 数据。
    // 输入输出名称/形状见 model_meta.json（AXMODEL 即按此编译）。
    std::vector<std::vector<float>> Run(const std::vector<std::vector<float>>& inputs);

private:
    struct Impl;
    Impl* impl_;
};
