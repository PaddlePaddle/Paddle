// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/layernorm_shift_partition_op.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
static void convertAndCopy(const std::vector<float> &host, T *dev) {
  T *host_ptr = new T[host.size()];
  std::transform(host.begin(), host.end(), host_ptr, [](float x) {
    return static_cast<T>(x);
  });
  cudaMemcpy(dev, host_ptr, sizeof(T) * host.size(), cudaMemcpyHostToDevice);
  delete host_ptr;
}

void LayernormShiftPartitionPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {
  int type_size = 0;
  switch (in->desc.type) {
    case nvinfer1::DataType::kHALF:
      type_size = sizeof(half);
      break;
    case nvinfer1::DataType::kFLOAT:
      type_size = sizeof(float);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Fatal("The LayernormShiftPartition TRT Plugin's "
                                  "input type should be float or half."));
  }

  if (gamma_dev_ == nullptr) {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    gamma_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (in->desc.type == nvinfer1::DataType::kHALF)
      convertAndCopy(gamma_, reinterpret_cast<half *>(p));
    else
      convertAndCopy(gamma_, reinterpret_cast<float *>(p));
  }

  if (beta_dev_ == nullptr) {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    beta_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (in->desc.type == nvinfer1::DataType::kHALF)
      convertAndCopy(beta_, reinterpret_cast<half *>(p));
    else
      convertAndCopy(beta_, reinterpret_cast<float *>(p));
  }
}

bool LayernormShiftPartitionPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument("The input of LayernormShiftPartition "
                                        "plugin shoule not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      bool res = (in.type == nvinfer1::DataType::kFLOAT ||
                  in.type == nvinfer1::DataType::kHALF);
      res = res && (in.format == nvinfer1::TensorFormat::kLINEAR);
      return res;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT;
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType LayernormShiftPartitionPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index,
      0,
      platform::errors::InvalidArgument(
          "The LayernormShiftPartition only has one input, so the "
          "index value should be 0, but get %d.",
          index));
  return input_types[0];
}

nvinfer1::DimsExprs LayernormShiftPartitionPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      output_index,
      0,
      platform::errors::InvalidArgument(
          "There is only one output of the LayernormShiftPartition, "
          "so the index should be zero,"
          "but it's (%d)",
          output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      platform::errors::InvalidArgument(
          "The Input of the LayernormShiftPartition should be 1, but we found "
          "it has (%d) inputs",
          nb_inputs));

  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = expr_builder.operation(
      nvinfer1::DimensionOperation::kFLOOR_DIV,
      *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                              *inputs[0].d[0],
                              *inputs[0].d[1]),
      *expr_builder.constant(window_size_ * window_size_));
  ret.d[1] = expr_builder.constant(window_size_ * window_size_);
  ret.d[2] = expr_builder.constant(shift_size_);
  return ret;
}

int LayernormShiftPartitionPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. LayernormShiftPartition-->fp32";

  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The LayerNorm TRT Plugin's input type should be float."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
