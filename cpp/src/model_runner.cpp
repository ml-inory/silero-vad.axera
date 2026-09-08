#include "model_runner.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {

std::vector<char> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::vector<char>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

void check_ax(int ret, const char* message) {
    if (ret != 0) {
        throw std::runtime_error(message);
    }
}

}  // namespace

struct ModelRunner::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_CONTEXT_T context = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io {};
    std::vector<AX_ENGINE_IO_BUFFER_T> inputs;
    std::vector<AX_ENGINE_IO_BUFFER_T> outputs;
    std::vector<char> model;

    explicit Impl(const std::string& model_path, const std::string& model_name)
        : model(read_binary(model_path)) {
        check_ax(AX_SYS_Init(), "AX_SYS_Init failed");
        AX_ENGINE_NPU_ATTR_T npu_attr;
        std::memset(&npu_attr, 0, sizeof(npu_attr));
        if (AX_ENGINE_GetVNPUAttr(&npu_attr) != 0) {
            npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        }
        check_ax(AX_ENGINE_Init(&npu_attr), "AX_ENGINE_Init failed");
        AX_ENGINE_HANDLE_EXTRA_T extra;
        std::memset(&extra, 0, sizeof(extra));
        extra.pName = const_cast<AX_S8*>(reinterpret_cast<const AX_S8*>(model_name.c_str()));
        check_ax(
            AX_ENGINE_CreateHandleV2(
                &handle, model.data(), static_cast<AX_U32>(model.size()), &extra),
            "AX_ENGINE_CreateHandleV2 failed");
        check_ax(AX_ENGINE_CreateContextV2(handle, &context), "AX_ENGINE_CreateContextV2 failed");
        check_ax(AX_ENGINE_GetIOInfo(handle, &info), "AX_ENGINE_GetIOInfo failed");
        if (!info || info->nInputSize < 1 || info->nOutputSize < 1) {
            throw std::runtime_error("model has no input or output tensors");
        }

        inputs.resize(info->nInputSize);
        outputs.resize(info->nOutputSize);
        io.pInputs = inputs.data();
        io.nInputSize = info->nInputSize;
        io.pOutputs = outputs.data();
        io.nOutputSize = info->nOutputSize;

        for (AX_U32 i = 0; i < info->nInputSize; ++i) {
            std::memset(&inputs[i], 0, sizeof(inputs[i]));
            inputs[i].nSize = info->pInputs[i].nSize;
            check_ax(
                AX_SYS_MemAllocCached(
                    &inputs[i].phyAddr, &inputs[i].pVirAddr, inputs[i].nSize, 128,
                    reinterpret_cast<const AX_S8*>("model_input")),
                "AX_SYS_MemAllocCached failed");
        }
        for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
            std::memset(&outputs[i], 0, sizeof(outputs[i]));
            outputs[i].nSize = info->pOutputs[i].nSize;
            check_ax(
                AX_SYS_MemAllocCached(
                    &outputs[i].phyAddr, &outputs[i].pVirAddr, outputs[i].nSize, 128,
                    reinterpret_cast<const AX_S8*>("model_output")),
                "AX_SYS_MemAllocCached failed");
        }
    }

    ~Impl() {
        for (auto& item : inputs) {
            if (item.phyAddr) AX_SYS_MemFree(item.phyAddr, item.pVirAddr);
        }
        for (auto& item : outputs) {
            if (item.phyAddr) AX_SYS_MemFree(item.phyAddr, item.pVirAddr);
        }
        if (handle) AX_ENGINE_DestroyHandle(handle);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
};

ModelRunner::ModelRunner(const std::string& model_path, const std::string& model_name)
    : impl_(new Impl(model_path, model_name)) {}

ModelRunner::~ModelRunner() {
    delete impl_;
}

size_t ModelRunner::NumInputs() const {
    return impl_->info->nInputSize;
}

size_t ModelRunner::NumOutputs() const {
    return impl_->info->nOutputSize;
}

size_t ModelRunner::InputBytes(size_t index) const {
    return impl_->inputs.at(index).nSize;
}

size_t ModelRunner::OutputBytes(size_t index) const {
    return impl_->outputs.at(index).nSize;
}

std::vector<std::vector<float>> ModelRunner::Run(const std::vector<std::vector<float>>& inputs) {
    if (inputs.size() != NumInputs()) {
        throw std::runtime_error("input count mismatch");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() * sizeof(float) > impl_->inputs[i].nSize) {
            throw std::runtime_error("input tensor is larger than model input buffer");
        }
        std::memcpy(impl_->inputs[i].pVirAddr, inputs[i].data(), inputs[i].size() * sizeof(float));
        AX_SYS_MflushCache(impl_->inputs[i].phyAddr, impl_->inputs[i].pVirAddr,
                           impl_->inputs[i].nSize);
    }
    check_ax(AX_ENGINE_RunSyncV2(impl_->handle, impl_->context, &impl_->io),
             "AX_ENGINE_RunSyncV2 failed");

    std::vector<std::vector<float>> outputs(NumOutputs());
    for (size_t i = 0; i < outputs.size(); ++i) {
        AX_SYS_MinvalidateCache(impl_->outputs[i].phyAddr, impl_->outputs[i].pVirAddr,
                                impl_->outputs[i].nSize);
        const size_t count = impl_->outputs[i].nSize / sizeof(float);
        const auto* src = static_cast<const float*>(impl_->outputs[i].pVirAddr);
        outputs[i].assign(src, src + count);
    }
    return outputs;
}
