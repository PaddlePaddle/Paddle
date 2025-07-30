/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
   Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#include "paddle/fluid/inference/tensorrt/plugin/skip_merge_layernorm_op_plugin.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define FINAL_MASK 0xffffffff

template <typename T>
__global__ void merge_layernorm_v2(T *out,
                                   const T *__restrict input0,
                                   const T *__restrict input1,
                                   const T *__restrict gamma,
                                   const T *__restrict beta,
                                   const float layernorm_eps,
                                   int batch,
                                   int H,
                                   int W,
                                   int n) {
  // input is [batch, 2*H, 2*W, n/4]
  // output is [batch, H, W, n]
  // grid (W, H, batch)
  // block (n)
  const int kIte = 4;
  const int tid = threadIdx.x;
  const int W_idx = blockIdx.x;
  const int H_idx = blockIdx.y;
  const size_t batch_offset = blockIdx.z * H * W * n;
  const int input_H_stride = W * n / 2;
  const int output_H_stride = W * n;
  const int n_4 = n >> 2;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float local_out[kIte];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      int part_id = col_id / n_4;
      int offset_in_W = part_id / 2;
      int offset_in_H = part_id % 2;
      size_t input_id = batch_offset +
                        (2 * H_idx + offset_in_H) * input_H_stride +
                        (2 * W_idx + offset_in_W) * n_4 + (col_id % n_4);
      local_out[i] = static_cast<float>(__ldg(input0 + input_id));
      local_out[i] += static_cast<float>(__ldg(input1 + input_id));
      sum += local_out[i];
    }
  }

  mean = phi::funcs::BlockReduceSum<float>(sum, FINAL_MASK);
  if (tid == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      local_out[i] = local_out[i] - s_mean;
      var += local_out[i] * local_out[i];
    }
  }

  variance = phi::funcs::BlockReduceSum<float>(var, FINAL_MASK);
  if (tid == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      size_t output_idx =
          batch_offset + H_idx * output_H_stride + W_idx * n + col_id;
      out[output_idx] =
          static_cast<T>(local_out[i] * s_variance *
                             static_cast<float>(__ldg(&gamma[col_id])) +
                         static_cast<float>(__ldg(&beta[col_id])));
    }
  }
}

template <typename T>
void invokeMergeLayernorm(T *output,
                          const T *input0,
                          const T *input1,
                          const T *gamma,
                          const T *beta,
                          float layernorm_eps,
                          int batch,
                          int H,
                          int W,
                          int n,
                          cudaStream_t stream) {
  if ((W % 2 != 0) || (H % 2 != 0)) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "H(W) of merge layernorm should be a multiple of 2."));
  }
  dim3 grid(W / 2, H / 2, batch);
  int blockSize = (n + 31) / 32 * 32;
  merge_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(output,
                                                        input0,
                                                        input1,
                                                        gamma,
                                                        beta,
                                                        layernorm_eps,
                                                        batch,
                                                        H / 2,
                                                        W / 2,
                                                        n * 4);
}

template void invokeMergeLayernorm<float>(float *output,
                                          const float *input0,
                                          const float *input1,
                                          const float *gamma,
                                          const float *beta,
                                          float layernorm_eps,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          cudaStream_t stream);

template void invokeMergeLayernorm<half>(half *output,
                                         const half *input0,
                                         const half *input1,
                                         const half *gamma,
                                         const half *beta,
                                         float layernorm_eps,
                                         int batch,
                                         int H,
                                         int W,
                                         int n,
                                         cudaStream_t stream);

template <typename T>
static void convertAndCopy(const std::vector<float> &host, T *dev) {
  T *host_ptr = new T[host.size()];
  std::transform(host.begin(), host.end(), host_ptr, [](float x) {
    return static_cast<T>(x);
  });
  cudaMemcpy(dev, host_ptr, sizeof(T) * host.size(), cudaMemcpyHostToDevice);
  delete host_ptr;
}

void SkipMergeLayernormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

SkipMergeLayernormPluginDynamic::SkipMergeLayernormPluginDynamic(
    const float *bias_d,
    const size_t bias_num,
    const float *scale_d,
    const size_t scale_num,
    const float eps,
    const int begin_norm_axis,
    const bool with_fp16,
    std::shared_ptr<void> bias_device,
    std::shared_ptr<void> scale_device)
    : eps_(eps),
      begin_norm_axis_(begin_norm_axis),
      with_fp16_(with_fp16),
      bias_device_(bias_device),
      scale_device_(scale_device) {
  bias_.resize(bias_num);
  scale_.resize(scale_num);
  std::copy(bias_d, bias_d + bias_num, bias_.data());
  std::copy(scale_d, scale_d + scale_num, scale_.data());
  int type_size = with_fp16_ ? sizeof(half) : sizeof(float);
  if (bias_device_ == nullptr) {
    void *p;
    cudaMalloc(&p, bias_num * type_size);
    bias_device_.reset(p, [](void *ptr) { cudaFree(ptr); });

    if (with_fp16) {
      convertAndCopy<half>(bias_, reinterpret_cast<half *>(p));
    } else {
      convertAndCopy<float>(bias_, reinterpret_cast<float *>(p));
    }
  }
  if (scale_device_ == nullptr) {
    void *p;
    cudaMalloc(&p, scale_num * type_size);
    scale_device_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (with_fp16) {
      convertAndCopy<half>(scale_, reinterpret_cast<half *>(p));
    } else {
      convertAndCopy<float>(scale_, reinterpret_cast<float *>(p));
    }
  }
}

bool SkipMergeLayernormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument("The input of MergeLayernorm "
                                      "plugin should not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return in.type == nvinfer1::DataType::kHALF &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType SkipMergeLayernormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The MergeLayernorm only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

nvinfer1::DimsExprs SkipMergeLayernormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = expr_builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                    *inputs[0].d[1],
                                    *expr_builder.constant(4));
  ret.d[2] = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                    *inputs[0].d[2],
                                    *expr_builder.constant(4));
  return ret;
}

int SkipMergeLayernormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  int batch = input_dims.d[0];
  int input_resolution = static_cast<int>(std::sqrt(input_dims.d[1]));
  int dim = static_cast<int>(input_dims.d[2]);
  PADDLE_ENFORCE_EQ(
      input_resolution * input_resolution,
      input_dims.d[1],
      common::errors::InvalidArgument(
          "The MergeLayernorm TRT Plugin get invalid input_resolution %d",
          input_resolution));

  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. MergeLayernorm-->fp32";
    invokeMergeLayernorm<float>(
        reinterpret_cast<float *>(outputs[0]),
        reinterpret_cast<const float *>(inputs[0]),
        reinterpret_cast<const float *>(inputs[1]),
        reinterpret_cast<const float *>(scale_device_.get()),
        reinterpret_cast<const float *>(bias_device_.get()),
        eps_,
        batch,
        input_resolution,
        input_resolution,
        dim,
        stream);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. MergeLayernorm-->fp16";
    invokeMergeLayernorm<half>(
        reinterpret_cast<half *>(outputs[0]),
        reinterpret_cast<const half *>(inputs[0]),
        reinterpret_cast<const half *>(inputs[1]),
        reinterpret_cast<const half *>(scale_device_.get()),
        reinterpret_cast<const half *>(bias_device_.get()),
        eps_,
        batch,
        input_resolution,
        input_resolution,
        dim,
        stream);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The MergeLayernorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
