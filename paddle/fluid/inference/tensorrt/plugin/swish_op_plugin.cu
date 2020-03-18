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

#include <stdio.h>
#include <cassert>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/plugin_kernel_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/swish_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
T *CreateSwishPluginDeserialize(const void *buffer, size_t length) {
  return new T(buffer, length);
}
REGISTER_TRT_PLUGIN("swish_plugin", PluginTensorRT,
                    CreateSwishPluginDeserialize<SwishPlugin>);

int SwishPlugin::initialize() { return 0; }

nvinfer1::Dims SwishPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T>
__global__ void swish_kernel(int num, const T *input, T *output, T beta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
#if __CUDA_ARCH__ >= 350
    output[index] =
        __ldg(input + index) /
        (static_cast<T>(1.0) + exp_func<T>(-beta * __ldg(input + index)));
#else
    output[index] = input[index] /
                    (static_cast<T>(1.0) + exp_func<T>(-beta * input[index]));
#endif
  }
}

size_t SwishPlugin::getSerializationSize() {
  return getBaseSerializationSize() + SerializedSize(beta_) +
         SerializedSize(getPluginType());
}

void SwishPlugin::serialize(void *buffer) {
  SerializeValue(&buffer, getPluginType());
  serializeBase(buffer);
  SerializeValue(&buffer, beta_);
}

int SwishPlugin::enqueue(int batch_size, const void *const *inputs,
                         void **outputs, void *workspace, cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;

  auto type = getDataType();
  if (type == nvinfer1::DataType::kFLOAT) {
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = reinterpret_cast<float **>(outputs)[0];
    swish_kernel<float><<<blocks, threads, 0, stream>>>(num, input, output,
                                                        beta_);
  } else if (type == nvinfer1::DataType::kHALF) {
#ifdef SUPPORT_CUDA_FP16
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = reinterpret_cast<half **>(outputs)[0];
    half beta = static_cast<half>(beta_);
    swish_kernel<half><<<blocks, threads, 0, stream>>>(num, input, output,
                                                       beta_);
#else
    PADDLE_THROW("The cuda arch must greater than 600.");
#endif
  } else {
    PADDLE_THROW("The Swish TRT Plugin's input type should be float or half.");
  }
  // input dims is CHW.
  return cudaGetLastError() != cudaSuccess;
}

bool SwishPlugin::supportsFormat(nvinfer1::DataType type,
                                 nvinfer1::PluginFormat format) const {
#ifdef SUPPORT_CUDA_FP16
  return ((type == nvinfer1::DataType::kFLOAT ||
           type == nvinfer1::DataType::kHALF) &&
          (format == nvinfer1::PluginFormat::kNCHW));
#else
  return ((type == nvinfer1::DataType::kFLOAT) &&
          (format == nvinfer1::PluginFormat::kNCHW));
#endif
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

int SwishPluginDynamic::initialize() { return 0; }

size_t SwishPluginDynamic::getSerializationSize() const {
  return getBaseSerializationSize() + SerializedSize(beta_) +
         SerializedSize(getPluginType());
}

void SwishPluginDynamic::serialize(void *buffer) const {
  SerializeValue(&buffer, getPluginType());
  serializeBase(buffer);
  SerializeValue(&buffer, beta_);
}

nvinfer1::DimsExprs SwishPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  return inputs[0];
}

bool SwishPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

#ifdef SUPPORT_CUDA_FP16
  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT ||
           in_out[pos].type == nvinfer1::DataType::kHALF) &&
          in_out[pos].format == nvinfer1::PluginFormat::kNCHW);
#else
  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kNCHW);
#endif
}

nvinfer1::DataType SwishPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The Swish Plugin only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true, platform::errors::InvalidArgument(
                              "The input type should be half or float"));
  return input_types[0];
}

int SwishPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                                const nvinfer1::PluginTensorDesc *output_desc,
                                const void *const *inputs, void *const *outputs,
                                void *workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    const float *input = static_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    swish_kernel<float><<<blocks, threads, 0, stream>>>(num, input, output,
                                                        beta_);
  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef SUPPORT_CUDA_FP16
    const half *input = static_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    half beta = static_cast<half>(beta);
    swish_kernel<half><<<blocks, threads, 0, stream>>>(num, input, output,
                                                       beta);
#else
    PADDLE_THROW("The cuda arch must greater than 600.");
#endif
  } else {
    PADDLE_THROW("The Swish TRT Plugin's input type should be float or half.");
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
