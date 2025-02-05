// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__global__ void layernorm_shift_partition(T *out,
                                          const T *input,
                                          const T *gamma,
                                          const T *beta,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          int shift_size,
                                          int window_size,
                                          const float eps) {
  int tid = threadIdx.x;
  const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
  const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
  const int shifted_H_idx =
      (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y)
                        : blockIdx.y;
  const int shifted_W_idx =
      (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x)
                        : blockIdx.x;
  const int window_H_idx = shifted_H_idx / window_size;
  const int window_W_idx = shifted_W_idx / window_size;
  const int stride_of_window_H = W / window_size;
  const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
  const int idx_in_window = (shifted_H_idx % window_size) * window_size +
                            (shifted_W_idx % window_size);
  const int output_bid =
      batch_offset + window_idx * window_size * window_size + idx_in_window;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  float local_out =
      (tid < n) ? static_cast<float>(__ldg(input + bid * n + tid)) : 0.0f;
#else
  float local_out = (tid < n) ? static_cast<float>(input[bid * n + tid]) : 0.0f;
#endif

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float diff = (tid < n) ? (local_out - s_mean) : 0.0f;
  variance = blockReduceSum<float>(diff * diff);
  if (threadIdx.x == 0) {
    s_variance = variance / n + eps;
  }
  __syncthreads();

  if (tid < n) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
    out[output_bid * n + tid] =
        (T)(((local_out - s_mean) * rsqrtf(s_variance)) *
                static_cast<float>(__ldg(&gamma[tid])) +
            static_cast<float>(__ldg(&beta[tid])));
#else
    out[output_bid * n + tid] =
        (T)(((local_out - s_mean) * rsqrtf(s_variance)) *
                static_cast<float>(gamma[tid]) +
            static_cast<float>(beta[tid]));
#endif
  }
}

template <>
__global__ void layernorm_shift_partition(half2 *out_ptr,
                                          const half2 *input_ptr,
                                          const half2 *gamma_ptr,
                                          const half2 *beta_ptr,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          int shift_size,
                                          int window_size,
                                          const float eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
  const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
  const int shifted_H_idx =
      (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y)
                        : blockIdx.y;
  const int shifted_W_idx =
      (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x)
                        : blockIdx.x;
  const int window_H_idx = shifted_H_idx / window_size;
  const int window_W_idx = shifted_W_idx / window_size;
  const int stride_of_window_H = W / window_size;
  const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
  const int idx_in_window = (shifted_H_idx % window_size) * window_size +
                            (shifted_W_idx % window_size);
  const int output_bid =
      batch_offset + window_idx * window_size * window_size + idx_in_window;
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  float local_out = 0.0f;
  int id = bid * n + tid;
  if (tid < n) {
    local_out_fp2 = __half22float2(__ldg(input_ptr + id));
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;
  }

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) {
    s_mean = mean / (n * 2);
  }
  __syncthreads();

  if (tid < n) {
    variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
    variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (n * 2) + eps);
  }
  __syncthreads();

  if (tid < n) {
    float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
    float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
    local_out_fp2.x =
        (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y =
        (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    out_ptr[output_bid * n + tid] = __float22half2_rn(local_out_fp2);
  }
#endif
}

#define kITE 4
template <typename T>
__global__ void layernorm_shift_partition_v2(T *out,
                                             const T *__restrict input,
                                             const T *__restrict gamma,
                                             const T *__restrict beta,
                                             int batch,
                                             int H,
                                             int W,
                                             int n,
                                             int shift_size,
                                             int window_size,
                                             const float eps) {
  // constexpr int kITE = 4;
  const int tid = threadIdx.x;
  const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
  const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
  const int shifted_H_idx =
      (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y)
                        : blockIdx.y;
  const int shifted_W_idx =
      (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x)
                        : blockIdx.x;
  const int window_H_idx = shifted_H_idx / window_size;
  const int window_W_idx = shifted_W_idx / window_size;
  const int stride_of_window_H = W / window_size;
  const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
  const int idx_in_window = (shifted_H_idx % window_size) * window_size +
                            (shifted_W_idx % window_size);
  const int output_bid =
      batch_offset + window_idx * window_size * window_size + idx_in_window;
  const int offset = bid * n;
  const int output_offset = output_bid * n;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float local_out[kITE];

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
      local_out[i] = static_cast<float>(__ldg(input + offset + col_id));
#else
      local_out[i] = static_cast<float>(input[offset + col_id]);
#endif
      sum += local_out[i];
    }
  }

  mean = blockReduceSum<float>(sum);
  if (tid == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      float diff = local_out[i] - s_mean;
      local_out[i] = diff;
      var += diff * diff;
    }
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0) {
    s_variance = rsqrtf(variance / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
      out[output_offset + col_id] =
          (T)(local_out[i] * s_variance *
                  static_cast<float>(__ldg(&gamma[col_id])) +
              static_cast<float>(__ldg(&beta[col_id])));
#else
      out[output_offset + col_id] =
          (T)(local_out[i] * s_variance * static_cast<float>(gamma[col_id]) +
              static_cast<float>(beta[col_id]));
#endif
    }
  }
}

template <>
__global__ void layernorm_shift_partition_v2(half2 *out_ptr,
                                             const half2 *__restrict input_ptr,
                                             const half2 *__restrict gamma_ptr,
                                             const half2 *__restrict beta_ptr,
                                             int batch,
                                             int H,
                                             int W,
                                             int n,
                                             int shift_size,
                                             int window_size,
                                             const float eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  // constexpr int ite = 4;
  const int tid = threadIdx.x;
  const int batch_offset = blockIdx.z * gridDim.y * gridDim.x;
  const int bid = batch_offset + blockIdx.y * gridDim.x + blockIdx.x;
  const int shifted_H_idx =
      (shift_size != 0) ? ((blockIdx.y - shift_size + gridDim.y) % gridDim.y)
                        : blockIdx.y;
  const int shifted_W_idx =
      (shift_size != 0) ? ((blockIdx.x - shift_size + gridDim.x) % gridDim.x)
                        : blockIdx.x;
  const int window_H_idx = shifted_H_idx / window_size;
  const int window_W_idx = shifted_W_idx / window_size;
  const int stride_of_window_H = W / window_size;
  const int window_idx = window_H_idx * stride_of_window_H + window_W_idx;
  const int idx_in_window = (shifted_H_idx % window_size) * window_size +
                            (shifted_W_idx % window_size);
  const int output_bid =
      batch_offset + window_idx * window_size * window_size + idx_in_window;
  const int offset = bid * n;
  const int output_offset = output_bid * n;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  half2 local_out_half2[kITE];
  const half2 zero = {static_cast<half>(0.0f), static_cast<half>(0.0f)};

  // float sum = 0.0f;
  half2 sum = __float2half2_rn(0.0f);
#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      local_out_half2[i] = __ldg(input_ptr + offset + col_id);
      sum += local_out_half2[i];
    }
  }

  mean = blockReduceSum<float>(static_cast<float>(sum.x + sum.y));
  if (threadIdx.x == 0) {
    s_mean = mean / (n * 2);
  }
  __syncthreads();

  float var = 0.0f;
  half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      local_out_half2[i] = local_out_half2[i] - s_mean_2;
      float v1 = static_cast<float>(local_out_half2[i].x);
      float v2 = static_cast<float>(local_out_half2[i].y);
      var += v1 * v1 + v2 * v2;
    }
  }

  variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (n * 2) + eps);
  }
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
  for (int i = 0; i < kITE; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      out_ptr[output_offset + col_id] =
          local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) +
          __ldg(&beta_ptr[col_id]);
    }
  }
#endif
}

template <typename T>
void invokeLayernormShiftPartition(T *out,
                                   const T *input,
                                   const T *gamma,
                                   const T *beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n,
                                   int shift_size,
                                   int window_size,
                                   const float eps,
                                   cudaStream_t stream) {
  dim3 grid(W, H, batch);
  int blockSize = (n + 31) / 32 * 32;
  if (blockSize >= 768) {
    blockSize = ((blockSize / 4) + 31) / 32 * 32;
    layernorm_shift_partition_v2<T><<<grid, blockSize, 0, stream>>>(
        out, input, gamma, beta, batch, H, W, n, shift_size, window_size, eps);
  } else {
    layernorm_shift_partition<T><<<grid, blockSize, 0, stream>>>(
        out, input, gamma, beta, batch, H, W, n, shift_size, window_size, eps);
  }
}

template <>
void invokeLayernormShiftPartition(half *out,
                                   const half *input,
                                   const half *gamma,
                                   const half *beta,
                                   int batch,
                                   int H,
                                   int W,
                                   int n,
                                   int shift_size,
                                   int window_size,
                                   const float eps,
                                   cudaStream_t stream) {
  dim3 grid(W, H, batch);
  int blockSize = n / 2;
  blockSize = (blockSize + 31) / 32 * 32;

  if ((batch * H * W >= 512 && blockSize >= 768) || blockSize > 1024) {
    blockSize = ((blockSize / 4) + 31) / 32 * 32;
    layernorm_shift_partition_v2<<<grid, blockSize, 0, stream>>>(
        reinterpret_cast<half2 *>(out),
        (const half2 *)input,
        (const half2 *)gamma,
        (const half2 *)beta,
        batch,
        H,
        W,
        n / 2,
        shift_size,
        window_size,
        eps);
  } else {
    layernorm_shift_partition<<<grid, blockSize, 0, stream>>>(
        reinterpret_cast<half2 *>(out),
        (const half2 *)input,
        (const half2 *)gamma,
        (const half2 *)beta,
        batch,
        H,
        W,
        n / 2,
        shift_size,
        window_size,
        eps);
  }
}

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
    int nbOutputs) TRT_NOEXCEPT {}

LayernormShiftPartitionPluginDynamic::LayernormShiftPartitionPluginDynamic(
    const float *gamma,
    const float *beta,
    const int param_num,
    int shift_size,
    int window_size,
    int input_resolution,
    float eps,
    bool with_fp16,
    std::shared_ptr<void> gamma_dev,
    std::shared_ptr<void> beta_dev)
    : with_fp16_(with_fp16),
      window_size_(window_size),
      shift_size_(shift_size),
      input_resolution_(input_resolution),
      eps_(eps),
      param_num_(param_num),
      gamma_dev_(gamma_dev),
      beta_dev_(beta_dev) {
  beta_.resize(param_num);
  gamma_.resize(param_num);
  std::copy(gamma, gamma + param_num, gamma_.data());
  std::copy(beta, beta + param_num, beta_.data());
  int type_size = with_fp16 ? sizeof(half) : sizeof(float);
  if (gamma_dev_ == nullptr) {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    gamma_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (with_fp16)
      convertAndCopy(gamma_, reinterpret_cast<half *>(p));
    else
      convertAndCopy(gamma_, reinterpret_cast<float *>(p));
  }
  if (beta_dev_ == nullptr) {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    beta_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (with_fp16)
      convertAndCopy(beta_, reinterpret_cast<half *>(p));
    else
      convertAndCopy(beta_, reinterpret_cast<float *>(p));
  }
}

LayernormShiftPartitionPluginDynamic::LayernormShiftPartitionPluginDynamic(
    void const *serialData, size_t serialLength) {
  DeserializeValue(&serialData, &serialLength, &beta_);
  DeserializeValue(&serialData, &serialLength, &gamma_);
  DeserializeValue(&serialData, &serialLength, &param_num_);
  DeserializeValue(&serialData, &serialLength, &with_fp16_);
  DeserializeValue(&serialData, &serialLength, &shift_size_);
  DeserializeValue(&serialData, &serialLength, &window_size_);
  DeserializeValue(&serialData, &serialLength, &input_resolution_);
  DeserializeValue(&serialData, &serialLength, &eps_);
  int type_size = with_fp16_ ? sizeof(half) : sizeof(float);
  {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    gamma_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (with_fp16_)
      convertAndCopy(gamma_, reinterpret_cast<half *>(p));
    else
      convertAndCopy(gamma_, reinterpret_cast<float *>(p));
  }
  {
    void *p;
    cudaMalloc(reinterpret_cast<void **>(&p), param_num_ * type_size);
    beta_dev_.reset(p, [](void *ptr) { cudaFree(ptr); });
    if (with_fp16_)
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
      common::errors::InvalidArgument("The input of LayernormShiftPartition "
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

nvinfer1::DataType LayernormShiftPartitionPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index,
      0,
      common::errors::InvalidArgument(
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
      common::errors::InvalidArgument(
          "There is only one output of the LayernormShiftPartition, "
          "so the index should be zero,"
          "but it's (%d)",
          output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      common::errors::InvalidArgument(
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
  ret.d[2] = inputs[0].d[2];
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
  int batch = input_dims.d[0];
  int emb_dim = input_dims.d[2];
  PADDLE_ENFORCE_EQ(
      input_resolution_ * input_resolution_,
      input_dims.d[1],
      common::errors::InvalidArgument(
          "The LayernormShiftPartition‘s input_resolution is wrong (%d)",
          input_dims.d[1]));
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. LayernormShiftPartition-->fp32";
    invokeLayernormShiftPartition(
        reinterpret_cast<float *>(outputs[0]),
        reinterpret_cast<const float *>(inputs[0]),
        reinterpret_cast<const float *>(gamma_dev_.get()),
        reinterpret_cast<const float *>(beta_dev_.get()),
        batch,
        input_resolution_,
        input_resolution_,
        emb_dim,
        shift_size_,
        window_size_,
        eps_,
        stream);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. LayernormShiftPartition-->half";
    invokeLayernormShiftPartition(
        reinterpret_cast<half *>(outputs[0]),
        reinterpret_cast<const half *>(inputs[0]),
        reinterpret_cast<const half *>(gamma_dev_.get()),
        reinterpret_cast<const half *>(beta_dev_.get()),
        batch,
        input_resolution_,
        input_resolution_,
        emb_dim,
        shift_size_,
        window_size_,
        eps_,
        stream);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The LayerNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
