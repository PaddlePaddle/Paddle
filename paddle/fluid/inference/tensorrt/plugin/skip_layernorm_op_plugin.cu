// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/skip_layernorm_op_plugin.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
void SkipLayerNormPluginDynamicImpl<T>::shareGPUData(
    const SkipLayerNormPluginDynamicImplBase *anthor) {
  auto *ptr = dynamic_cast<const SkipLayerNormPluginDynamicImpl<T> *>(anthor);
  if (!ptr->is_initialized_) {
    return;
  }
  scale_gpu_ = ptr->scale_gpu_;
  bias_gpu_ = ptr->bias_gpu_;
}

template <typename T>
int SkipLayerNormPluginDynamicImpl<T>::initialize() {
  if (is_initialized_) {
    return 0;
  }

  if (bias_) {
    cudaMalloc(&bias_gpu_, sizeof(T) * bias_size_);
    cudaMemcpy(
        bias_gpu_, bias_, bias_size_ * sizeof(T), cudaMemcpyHostToDevice);
  }
  if (scale_) {
    cudaMalloc(&scale_gpu_, sizeof(T) * scale_size_);
    cudaMemcpy(
        scale_gpu_, scale_, scale_size_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  is_initialized_ = true;
  return 0;
}

template <typename T>
void SkipLayerNormPluginDynamicImpl<T>::terminate() {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }

  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
}

int SkipLayerNormPluginDynamic::initialize() TRT_NOEXCEPT {
  impl_->initialize();

  return 0;
}

void SkipLayerNormPluginDynamic::terminate() TRT_NOEXCEPT {
  impl_->terminate();
}

nvinfer1::DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of swish plugin shoule not be nullptr."));
  PADDLE_ENFORCE_EQ(nb_outputs,
                    1,
                    platform::errors::InvalidArgument(
                        "The SkipLayerNorm's output should be one"
                        "but it's (%d) outputs.",
                        nb_outputs));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &desc = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      return (desc.type == nvinfer1::DataType::kHALF) &&
             (desc.format == nvinfer1::TensorFormat::kLINEAR);
#else
      return (desc.type == nvinfer1::DataType::kFLOAT) &&
             (desc.format == nvinfer1::TensorFormat::kLINEAR);
#endif
    } else {
      return (desc.type == nvinfer1::DataType::kFLOAT) &&
             (desc.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  if (pos == 1) {
    return desc.type == prev.type && desc.format == prev.format;
  }
  // output
  return desc.type == prev.type && desc.format == prev.format;
}

nvinfer1::DataType SkipLayerNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    platform::errors::InvalidArgument(
                        "The SkipLayerNorm Plugin only has one output, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}

template <typename T>
int SkipLayerNormPluginDynamicImpl<T>::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  int hidden = input_dims.d[2];

  auto input_type = input_desc[0].type;

  if (std::is_same<T, float>::value) {
    PADDLE_ENFORCE_EQ(input_type == nvinfer1::DataType::kFLOAT,
                      true,
                      platform::errors::InvalidArgument(
                          "The SkipLayernorm Plugin only support fp32 input."));
  } else if (std::is_same<T, half>::value) {
    PADDLE_ENFORCE_EQ(input_type == nvinfer1::DataType::kHALF,
                      true,
                      platform::errors::InvalidArgument(
                          "The SkipLayernorm Plugin only support fp16 input."));
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "Unsupport data type, the out type of SkipLayernorm should be "
        "float or half."));
  }
  auto *output_d = reinterpret_cast<T *>(outputs[0]);

  const T *input1 = reinterpret_cast<const T *>(inputs[0]);
  const T *input2 = reinterpret_cast<const T *>(inputs[1]);
  auto *output = reinterpret_cast<T *>(outputs[0]);
  operators::math::SkipLayerNormFunctor<T> skip_layer_norm_func;
  skip_layer_norm_func(
      num, hidden, input1, input2, scale_gpu_, bias_gpu_, output, eps_, stream);

  return cudaGetLastError() != cudaSuccess;
}

int SkipLayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  impl_->enqueue(input_desc, output_desc, inputs, outputs, workspace, stream);
  return cudaGetLastError() != cudaSuccess;
}

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
