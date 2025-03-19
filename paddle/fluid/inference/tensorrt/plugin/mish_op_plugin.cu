// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/mish_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

int MishPlugin::initialize() TRT_NOEXCEPT { return 0; }

bool MishPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
}

nvinfer1::Dims MishPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims* in_dims,
                                               int nb_inputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      common::errors::InvalidArgument("We expect [number of inputs] == 1"
                                      "in TRT Mish op plugin, but got "
                                      "[number of inputs] = %d.",
                                      nb_inputs));
  PADDLE_ENFORCE_LT(
      index,
      this->getNbOutputs(),
      common::errors::InvalidArgument("We expect [index] < [number of outputs]"
                                      "in TRT Mish op plugin, but got "
                                      "[index] = %d, [number of outputs] = %d.",
                                      index,
                                      this->getNbOutputs()));
  nvinfer1::Dims const& input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T>
__device__ T kTanh(T x) {
  return tanh(x);
}

template <>
__device__ half kTanh<half>(half x) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const float tmp = tanhf(__half2float(x));
  return __float2half(tmp);
#endif
}

template <typename T>
__device__ T kSoftplus(T x, T threshold) {
  return x > threshold ? x : log(exp(x) + static_cast<T>(1.0f));
}

template <>
__device__ half kSoftplus<half>(half x, half threshold) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  return x > threshold ? x : hlog(hexp(x) + static_cast<half>(1.0f));
#endif
}

template <typename T>
__global__ void mish_kernel(float threshold, int n, const T* input, T* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const T in = input[idx];
    output[idx] = in * kTanh<T>(kSoftplus<T>(in, static_cast<T>(threshold)));
  }
}

template <>
__global__ void mish_kernel<half>(float threshold,
                                  int n,
                                  const half* input,
                                  half* output) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const half in = input[idx];
    output[idx] =
        in * kTanh<half>(kSoftplus<half>(in, static_cast<half>(threshold)));
  }
#endif
}

#if IS_TRT_VERSION_LT(8000)
int MishPlugin::enqueue(int batchSize,
                        const void* const* inputs,
                        void** outputs,
#else
int MishPlugin::enqueue(int batchSize,
                        const void* const* inputs,
                        void* const* outputs,
#endif
                        void* workspace,
                        cudaStream_t stream) TRT_NOEXCEPT {
  const auto& input_dims = this->getInputDims(0);
  int num = batchSize;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }

  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  auto type = getDataType();
  if (type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. Mish-->fp32";
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    mish_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(threshold_, num, input, output);
  } else if (type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. Mish-->fp16";
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    mish_kernel<half>
        <<<grid_size, block_size, 0, stream>>>(threshold_, num, input, output);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The Mish TRT Plugin's input type should be float or half."));
  }

  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
int MishPluginDynamic::initialize() TRT_NOEXCEPT {
  getPluginNamespace();
  return 0;
}

size_t MishPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(threshold_) + SerializedSize(with_fp16_);
}

void MishPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, threshold_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs MishPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

bool MishPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of mish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return (in.type == nvinfer1::DataType::kFLOAT ||
              in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType MishPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The Mish Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int MishPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                               const nvinfer1::PluginTensorDesc* output_desc,
                               const void* const* inputs,
                               void* const* outputs,
                               void* workspace,
                               cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. Mish-->fp32";
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    mish_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(threshold_, num, input, output);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. Mish-->fp16";
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    mish_kernel<half>
        <<<grid_size, block_size, 0, stream>>>(threshold_, num, input, output);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The Mish TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
