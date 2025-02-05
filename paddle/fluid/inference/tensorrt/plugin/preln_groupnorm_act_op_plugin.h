/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/phi/kernels/group_norm_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
using phi::GroupNormNDHWCParams;
class PrelnGroupnormActPluginDynamic : public DynamicPluginTensorRT {
 public:
  PrelnGroupnormActPluginDynamic(const float* scale,
                                 const int scale_num,
                                 const float* bias,
                                 const int bias_num,
                                 float eps,
                                 int groups,
                                 bool with_silu,
                                 bool with_fp16,
                                 std::shared_ptr<void> scale_gpu = nullptr,
                                 std::shared_ptr<void> bias_gpu = nullptr)
      : scale_gpu_(scale_gpu),
        bias_gpu_(bias_gpu),
        groups_(groups),
        eps_(eps),
        with_silu_(with_silu),
        with_fp16_(with_fp16) {
    scale_.resize(scale_num);
    bias_.resize(bias_num);
    std::copy(scale, scale + scale_num, scale_.data());
    std::copy(bias, bias + bias_num, bias_.data());
    if (scale_gpu_ == nullptr) {
      void* p;
      cudaMalloc(reinterpret_cast<void**>(&p), scale_num * sizeof(float));
      scale_gpu_.reset(p, [](void* ptr) { cudaFree(ptr); });
      cudaMemcpy(
          p, scale_.data(), scale_num * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (bias_gpu_ == nullptr) {
      void* p;
      cudaMalloc(reinterpret_cast<void**>(&p), bias_num * sizeof(float));
      bias_gpu_.reset(p, [](void* ptr) { cudaFree(ptr); });
      cudaMemcpy(
          p, bias_.data(), bias_num * sizeof(float), cudaMemcpyHostToDevice);
    }
  }

  PrelnGroupnormActPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &groups_);
    DeserializeValue(&serialData, &serialLength, &with_silu_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);

    {
      void* p;
      cudaMalloc(reinterpret_cast<void**>(&p), scale_.size() * sizeof(float));
      scale_gpu_.reset(p, [](void* ptr) { cudaFree(ptr); });
      cudaMemcpy(p,
                 scale_.data(),
                 scale_.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    {
      void* p;
      cudaMalloc(reinterpret_cast<void**>(&p), bias_.size() * sizeof(float));
      bias_gpu_.reset(p, [](void* ptr) { cudaFree(ptr); });
      cudaMemcpy(p,
                 bias_.data(),
                 bias_.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
    }
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto* ptr = new PrelnGroupnormActPluginDynamic(scale_.data(),
                                                   scale_.size(),
                                                   bias_.data(),
                                                   bias_.size(),
                                                   eps_,
                                                   groups_,
                                                   with_silu_,
                                                   with_fp16_,
                                                   scale_gpu_,
                                                   bias_gpu_);
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "preln_groupnorm_act_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 2; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(scale_) + SerializedSize(bias_) +
           SerializedSize(eps_) + SerializedSize(groups_) +
           SerializedSize(with_silu_) + SerializedSize(with_fp16_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, groups_);
    SerializeValue(&buffer, with_silu_);
    SerializeValue(&buffer, with_fp16_);
  }
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override {
    // sizeof(float2) * maxBatchSize * maxNumberOfGroup. float2
    // contains two buffers for sum and squared sum;
    ws_ = sizeof(float) * 2 * in[0].max.d[0] * groups_;
  }

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return ws_;
  }
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
  void terminate() TRT_NOEXCEPT override{};

 private:
  size_t ws_;
  std::vector<float> scale_;
  std::vector<float> bias_;
  std::shared_ptr<void> scale_gpu_;
  std::shared_ptr<void> bias_gpu_;
  GroupNormNDHWCParams<__half> params_;
  int groups_;
  float eps_;
  bool with_silu_;
  bool with_fp16_;
};

class PrelnGroupnormActPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "preln_groupnorm_act_plugin_dynamic";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new PrelnGroupnormActPluginDynamic(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(PrelnGroupnormActPluginDynamicCreator);
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
