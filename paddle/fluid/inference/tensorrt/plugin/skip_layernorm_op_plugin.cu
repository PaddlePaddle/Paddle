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
#include "paddle/fluid/inference/tensorrt/plugin/plugin_kernel_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/skip_layernorm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
struct KeyValuePair;

template <>
struct KeyValuePair<float> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(float k, float v)
      : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  float key;
  float value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
    KeyValuePair tmp;
    tmp.key = key + a.key;
    tmp.value = value + a.value;
    return tmp;
  }
};

template <>
struct KeyValuePair<half> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(half k, half v) : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  half key;
  half value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
#if __CUDA_ARCH__ >= 600
    const half2 a2 = __halves2half2(key, value);
    const half2 b2 = __halves2half2(a.key, a.value);
    const half2 res = __hadd2(a2, b2);
    return KeyValuePair(res.x, res.y);
#else
    KeyValuePair tmp;
    tmp.key = FromFloat<half>(ToFloat<half>(key) + ToFloat<half>(a.key));
    tmp.value = FromFloat<half>(ToFloat<half>(value) + ToFloat<half>(a.value));
    return tmp;
#endif
  }
};

template <typename T>
using kvp = KeyValuePair<T>;

template <typename T, int TPB>
__device__ inline void layer_norm_small(T val, const kvp<T> &thread_data,
                                        const int ld, const int idx,
                                        const float *bias, const float *scale,
                                        T *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    const T g(scale[threadIdx.x]);
    const T b(bias[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormSmallKernel(int num, int hidden, const T *input1,
                                         const T *input2, T *output,
                                         const float *scale, const float *bias,
                                         float eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  kvp<T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, kvp<T>(rldval, rldval * val));
  }
  layer_norm_small<T, TPB>(val, thread_data, hidden, idx, bias, scale, output,
                           eps);
}

template <typename T, int TPB>
__device__ inline void layer_norm(const kvp<T> &thread_data, const int ld,
                                  const int offset, const float *bias,
                                  const float *scale, T *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(scale[i]);
    const T b(bias[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(int num, int hidden, const T *input1,
                                    const T *input2, T *output,
                                    const float *scale, const float *bias,
                                    float eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  kvp<T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const int idx = offset + it;
    const T val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, kvp<T>(rldval, rldval * val));
    output[idx] = val;
  }
  layer_norm<T, TPB>(thread_data, hidden, offset, bias, scale, output, eps);
}

template <typename T>
int ComputeLayerNorm(const int num, const int hidden, const T *input1,
                     const T *input2, const float *scale, const float *bias,
                     T *output, T eps, cudaStream_t stream) {
  int block = num / hidden;
  if (hidden <= 32) {
    const int threads = 32;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden <= 128) {
    const int threads = 128;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden == 384) {
    const int threads = 384;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else {
    const int threads = 256;
    SkipLayerNormKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  }
  return cudaGetLastError() != cudaSuccess;
}

int SkipLayerNormPluginDynamic::initialize() {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
  cudaMemcpy(bias_gpu_, bias_, bias_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
  cudaMemcpy(scale_gpu_, scale_, scale_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const {
  // return getBaseSerializationSize() + SerializedSize(beta_) +
  //       SerializedSize(getPluginType());
  return 0;
}

void SkipLayerNormPluginDynamic::serialize(void *buffer) const {
  //  SerializeValue(&buffer, getPluginType());
  //  serializeBase(buffer);
  //  SerializeValue(&buffer, beta_);
}

nvinfer1::DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  PADDLE_ENFORCE_EQ(
      inputs[0].nbDims, 5,
      platform::errors::InvalidArgument(
          "The Input dim of the SkipLayernorm should be 5, but it's (%d) now.",
          inputs[0].nbDims));
  return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
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

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
#ifdef SUPPORT_CUDA_FP16
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType SkipLayerNormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The SkipLayerNorm Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true, platform::errors::InvalidArgument(
                              "The input type should be half or float"));
  return input_types[0];
}

int SkipLayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  int hidden = input_dims.d[2];

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);
    float *output = static_cast<float *>(outputs[0]);
    ComputeLayerNorm<float>(num, hidden, input1, input2, scale_gpu_, bias_gpu_,
                            output, eps_, stream);
  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef SUPPORT_CUDA_FP16
    const half *input1 = static_cast<const half *>(inputs[0]);
    const half *input2 = static_cast<const half *>(inputs[1]);
    half *output = static_cast<half *>(outputs[0]);
    ComputeLayerNorm<half>(num, hidden, input1, input2, scale_gpu_, bias_gpu_,
                           output, static_cast<half>(eps_), stream);
#else
    PADDLE_THROW("The cuda arch must greater than 600.");
#endif
  } else {
    PADDLE_THROW(
        "The SkipLayerNorm TRT Plugin's input type should be float or half.");
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
