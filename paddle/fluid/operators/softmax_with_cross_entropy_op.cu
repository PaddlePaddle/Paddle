/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_cudnn_op.cu.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

// Wrapper of log function. Use log(float32) for float16
template <typename T>
static __device__ __forceinline__ T Log(T x) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  AccT logx = std::log(static_cast<AccT>(x));
  return math::TolerableValue<T>()(static_cast<T>(logx));
}

// Wrapper of exp function. Use exp(float32) for float16
template <typename T>
static __device__ __forceinline__ T Exp(T x) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  AccT expx = std::exp(static_cast<AccT>(x));
  return math::TolerableValue<T>()(static_cast<T>(expx));
}

// log2(value)
static inline int Log2Ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

enum class SoftmaxMode { kSoftmax, kLogSoftmax, kCrossEntropy };

/*
  Hard label cross entropy.
*/
template <typename T, bool IgnoreIndex>
__global__ void CrossEntropyHardLabel(T* loss, const T* softmax,
                                      const int64_t* labels, const int n,
                                      const int dim, const int d,
                                      const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;

  // thread ids compute loss[ids] using softmax[idx]
  if (ids < n * d) {
    if (labels[ids] < 0) {  // label is negative
      loss[ids] = static_cast<T>(0.0);
    } else {  // label is positive of zero
      int64_t idx = idx_n * dim * d + labels[ids] * d + idx_d;
      if (IgnoreIndex == true) {
        // IgnoreIndex is true
        if (labels[ids] == ignore_idx) {
          loss[ids] = static_cast<T>(0.0);
        } else {
          loss[ids] = -Log(softmax[idx]);
        }
      } else {
        // IgnoreIndex is false
        loss[ids] = -Log(softmax[idx]);
      }
    }
  }
}

/*
  Hard label cross entropy with exp.
  Input: log softmax
  Output: loss and exp(input)
*/
template <typename T, bool IgnoreIndex>
__global__ void CrossEntropyExpHardLabel(T* loss, T* softmax,
                                         const int64_t* labels, const int n,
                                         const int dim, const int d,
                                         const int ignore_idx) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    if (IgnoreIndex == true) {
      // IgnoreIndex is true
      if (idx_dim == labels[ids]) {
        if (labels[ids] == ignore_idx) {
          loss[ids] = static_cast<T>(0.0);
        } else {
          loss[ids] = -softmax[idx];
        }
      }
    } else {
      // IgnoreIndex is false
      if (labels[ids] >= 0 && labels[ids] < dim) {
        if (labels[ids] == idx_dim) {
          loss[ids] = -softmax[idx];
        }
      } else {
        loss[ids] = static_cast<T>(0.0);
      }
    }
    softmax[idx] = Exp(softmax[idx]);
  }
}

/*
  Core function of softmax with cross entropy forward
    - softmax, SoftmaxMode=kSoftmax
    - log softmax, SoftmaxMode=kLogSoftmax
    - softmax with cross entropy hard label, SoftmaxMode=kCrossEntropy
  The computation includes
    - Compute max value: maxvalue_{i} = max_j src_{i,j}
    - Compute sum of exp: s_{i} = sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
    - Compute: softmax_{i,j} = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    - Compute: logsoftmax_{i,j} = src_{i,j} - maxvalue_{i} - log(s_{i})
    - Compute: loss_{i} = -logsoftmax[i,label[i]] (Hard label)
  This computation results from following formula:
    softmax_{i,j} = e^{src_{i,j}} / sum_{j}{e^{src_{i,j}}}
                  = e^{src_{i,j} - maxvalue_{i}}
                    / sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
                  = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    logsoftmax_{i,j} = log(softmax_{i,j})
                     = src_{i,j} - maxvalue_{i} - log(s_{i})
  One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
  For reduction max (sum), firstly compute max (sum) to one warp, then use
  shuffle api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          SoftmaxMode mode, bool IgnoreIndex>
__global__ void WarpSoftmaxForward(T* loss, T* softmax, const T* src,
                                   const int64_t* label, const int batch_size,
                                   const int stride, const int element_count,
                                   const int ignore_index) {
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // read data from global memory
  AccT srcdata[kBatchSize][kIterationsV][kVSize];

#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
// read data to srcdata: - KVSize==1, - KVSize>1
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {
        if (src_idx < idx_max_v[i]) {
          srcdata[i][it][0] =
              static_cast<AccT>(src[(first_batch + i) * stride + src_idx]);
        } else {
          srcdata[i][it][0] = -std::numeric_limits<AccT>::infinity();
        }
      } else {
        const VecT* src_v =
            reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
        if (src_idx < idx_max_v[i]) {
          VecT srctmp = src_v[src_idx];
          const T* srcinptr = reinterpret_cast<const T*>(&srctmp);
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = static_cast<AccT>(srcinptr[s]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = -std::numeric_limits<AccT>::infinity();
          }
        }
      }
    }
  }

  // compute max value: maxvalue_{i} = max_j src_{i,j}
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    AccT valmax = srcdata[i][0][0];
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      valmax = (valmax > srcdata[i][0][s]) ? valmax : srcdata[i][0][s];
    }
    max_value[i] = valmax;

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
      AccT valmax = srcdata[i][it][0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcdata[i][it][s]) ? valmax : srcdata[i][it][s];
      }
      max_value[i] = (max_value[i] > valmax) ? max_value[i] : valmax;
    }
  }
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  AccT sum[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::exp(srcdata[i][0][0] - max_value[i]);
    } else {
      srcdata[i][0][0] = std::exp(srcdata[i][0][0] - max_value[i]);
      sum[i] = srcdata[i][0][0];
    }
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      if (mode == SoftmaxMode::kLogSoftmax ||
          mode == SoftmaxMode::kCrossEntropy) {
        sum[i] += std::exp(srcdata[i][0][s] - max_value[i]);
      } else {
        srcdata[i][0][s] = std::exp(srcdata[i][0][s] - max_value[i]);
        sum[i] += srcdata[i][0][s];
      }
    }

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (mode == SoftmaxMode::kLogSoftmax ||
            mode == SoftmaxMode::kCrossEntropy) {
          sum[i] += std::exp(srcdata[i][it][s] - max_value[i]);
        } else {
          srcdata[i][it][s] = std::exp(srcdata[i][it][s] - max_value[i]);
          sum[i] += srcdata[i][it][s];
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write data
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::log(sum[i]);
    }

#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {  // kVSize==1
        if (idx < idx_max_v[i]) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax
            softmax[(first_batch + i) * stride + idx] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize;
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (label[first_batch + i] == loss_idx) {
                if (label[first_batch + i] != ignore_index) {
                  loss[first_batch + i] = -logsoftmax;
                } else {
                  loss[first_batch + i] = static_cast<T>(0.0);
                }
              }
            } else {
              // IgnoreIndex is false
              if (label[first_batch + i] >= 0 &&
                  label[first_batch + i] < element_count) {
                if (label[first_batch + i] == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                loss[first_batch + i] = static_cast<T>(0.0);
              }
            }
          } else {  // softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] / sum[i];
          }
        } else {
          break;
        }
      } else {  // KVSize>1
        VecT* softmax_v =
            reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
        VecT tmpdata;
        T* tmpptr = reinterpret_cast<T*>(&tmpdata);
#pragma unroll
        for (int s = 0; s < kVSize; ++s) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            tmpptr[s] = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax
            tmpptr[s] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize + s;
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (label[first_batch + i] == loss_idx &&
                  label[first_batch + i] != ignore_index) {
                loss[first_batch + i] = -logsoftmax;
              }
            } else {
              // IgnoreIndex is false
              if (label[first_batch + i] >= 0 &&
                  label[first_batch + i] < element_count) {
                if (label[first_batch + i] == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                loss[first_batch + i] = static_cast<T>(0.0);
              }
            }
          } else {  // softmax
            tmpptr[s] = srcdata[i][it][s] / sum[i];
          }
        }
        if (idx < idx_max_v[i]) {
          softmax_v[idx] = tmpdata;
        } else {
          break;
        }
      }
    }
  }
}

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, VecT, AccT)           \
  case Log2Elements:                                                  \
    WarpSoftmaxForward<T, VecT, AccT, Log2Elements, mode,             \
                       IgnoreIndex><<<blocks, threads, 0, stream>>>(  \
        loss, softmax, src, label, batch_size, stride, element_count, \
        ignore_index);                                                \
    break;

/*
  Wrapper of softmax with cross entropy forward hard label.
*/
template <typename T, SoftmaxMode mode, bool IgnoreIndex>
void SwitchWarpSoftmaxForward(T* loss, T* softmax, const T* src,
                              const int64_t* label, const int batch_size,
                              const int stride, const int element_count,
                              const int ignore_index, gpuStream_t stream) {
  using AccT = typename details::MPTypeTrait<T>::Type;

  // use 128 threads per block to maximimize gpu utilization
  const int Log2Elements = static_cast<int>(Log2Ceil(element_count));
  const int kDimCeil = 1 << Log2Elements;
  int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / kWarpSize);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
  dim3 threads(kWarpSize, warps_per_block, 1);

  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, T, AccT);
    default:
      break;
  }
}

/*
  Wrapper of softmax with cross entropy hard label.
  - SwitchWarpSoftmaxForward for small size
  - cudnn function for large size
*/
template <typename T, bool IgnoreIndex>
static void SoftmaxWithCrossEntropyHardLabel(
    const platform::CUDADeviceContext& ctx, int rank, int axis,
    const T* logits_data, const int64_t* labels_data, T* loss_data,
    T* softmax_data, int N, int dim, int D, const int ignore_index) {
  auto stream = ctx.stream();
  constexpr int max_dim = 320;
  if (D == 1 && dim <= max_dim) {  // small size
    const SoftmaxMode mode = SoftmaxMode::kCrossEntropy;
    SwitchWarpSoftmaxForward<T, mode, IgnoreIndex>(
        loss_data, softmax_data, logits_data, labels_data, N, dim, dim,
        ignore_index, stream);
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
        handle, platform::CudnnDataType<T>::kOne(), descp, logits_data,
        platform::CudnnDataType<T>::kZero(), descp, softmax_data,
        MIOPEN_SOFTMAX_LOG, mode));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
        descp, logits_data, platform::CudnnDataType<T>::kZero(), descp,
        softmax_data));
#endif
    int threads = 128;
    int blocks = (N * dim * D + threads - 1) / threads;
    // compute cross entropy, input is log softmax
    CrossEntropyExpHardLabel<T, IgnoreIndex><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, labels_data, N, dim, D, ignore_index);
  }
}

/*
  Wrapper of softmax with cross entropy grad hard label.
*/
template <typename T>
__global__ void SoftmaxWithCrossEntropyGradHardLabel(
    T* logits_grad, const T* loss_grad, const int64_t* labels, const int64_t n,
    const int64_t dim, const int64_t d, const int ignore_index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    if (labels[ids] == ignore_index) {
      logits_grad[idx] = static_cast<T>(0.0);
    } else if (labels[ids] == idx_dim) {
      logits_grad[idx] =
          (logits_grad[idx] - static_cast<T>(1.0)) * loss_grad[ids];
    } else {
      logits_grad[idx] *= loss_grad[ids];
    }
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels, const int64_t n,
                                               const int64_t d,
                                               const int64_t remain) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int64_t idx_n = ids / d;
    int64_t idx_remain = ids % remain;
    int64_t idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
}

template <typename T>
__global__ void SoftLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const T* loss_grad,
                                                    const T* labels,
                                                    const int n, const int d,
                                                    const int remain) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int idx_n = ids / d;
    int idx_remain = ids % remain;
    int idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (-labels[ids] / logit_grad[ids]);
  }
}

template <typename T>
__global__ void HardLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const int64_t* labels,
                                                    const int n, const int d,
                                                    const int remain,
                                                    const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = labels[index];
    int idx = idx_n * d + tmp * remain + idx_remain;
    if (ignore_index != tmp) {
      logit_grad[idx] = -static_cast<T>(1.) / logit_grad[idx];
    }
  }
}

template <typename T>
__global__ void ScaleCrossEntropyGradient(T* logit_grad, const T* loss_grad,
                                          const int num, const int d,
                                          const int remain,
                                          const int64_t* labels,
                                          const int ignore_index) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    int idx_lbl = idx_n * remain + idx_remain;
    int k = (index % d) / remain;
    if (labels[idx_lbl] == ignore_index || labels[idx_lbl] != k) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

static __device__ __forceinline__ platform::float16 exp_on_device(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ platform::float16 log_on_device(
    platform::float16 x) {
  return math::TolerableValue<platform::float16>()(::Eigen::numext::log(x));
}
static __device__ __forceinline__ float log_on_device(float x) {
  return math::TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return math::TolerableValue<double>()(log(x));
}

/** In the following codes, 3 CUDA kernels are implemented to calculate softmax
 * and loss **/
/*
  Supposing the x is `logits` and y is `labels`, the equations are as
followings:
  cross\_entropy_i = \sum_{j}[- y_i_j * log({e^{x_i_j}/\sum_{j}e^{x_i_j}})]
        = \sum_{j}[- y_i_j * log({e^{x_i_j - max_i}/\sum_{j}e^{x_i_j-max_i}})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - log\sum_{j}e^{x_i_j - max_i})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - logDiffMaxSum_i)]
        = \sum_{j}(-y_i_j * tmp_i_j)
  softmax_i_j = e^{tmp_i_j}
where:
  max_i = \max_{j}{x_i_j}
  logDiffMaxSum_i = log\sum_{j}e^{x_i_j - max_i}
  tmp_i_j = x_i_j - max_i - logDiffMaxSum_i
Therefore, the calculation can be separated into 3 steps:
Step 1: row-wise operation to calculate max_i
Step 2: row-wise operation to calculate logDiffMaxSum_i
Step 3: calculate tmp_i_j, and finally get softmax_i_j and cross\_entropy_i
To save memory, we can share memory among max_i, logDiffMaxSum_i and
cross\_entropy_i.
In this way, the 3 steps should be changed to:
Step 1 (RowReductionForMax): row-wise operation to calculate max_i
Step 2 (RowReductionForDiffMaxSum): calculate immediate result of softmax'_i_j =
x_i_j - max_i, and row-wise operation to calculate logDiffMaxSum_i
Step 3 (RowReductionForSoftmaxAndCrossEntropy): calculate tmp_i_j = softmax'_i_j
- logDiffMaxSum_i, and finally get softmax_i_j and cross\_entropy_i
*/

// There are 3 kinds of reduce algorithms in cub:
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING
// BLOCK_REDUCE_WARP_REDUCTIONS (default)
template <typename T, int BlockDim>
using BlockReduce =
    cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

// Make sure that BlockDim <= axis_dim
// This kernel is used to calculate the max element of each row
template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int64_t d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits_data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / axis_dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  int64_t step = BlockDim * remain;
  T cur_max = logits_data[beg_idx];
  beg_idx += step;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += step;
  }

  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());

  if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
                                                 T* max_data, T* softmax,
                                                 int64_t d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / axis_dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int64_t step = BlockDim * remain;

  // In numeric stable mode softmax_with_loss, we calc loss with
  // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
  // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
  // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
  // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
  // So there is no need to clip on shift_softmax.
  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);

  if (!CalculateLogSoftmax) return;
  __syncthreads();
  diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }

  // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
  // calculate diff_max_sum, __syncthreads() is needed here.
  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}

#ifdef __HIPCC__  // @{ HIP Seperate Kernel for RowReductionForDiffMaxSum
// Note(qili93): HIP do not support return in kernel, need to seperate
// RowReductionForDiffMaxSum into two kernels below
template <typename T, int BlockDim>
static __global__ void RowReductionForSum(const T* logits_data, T* max_data,
                                          T* softmax, int64_t d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  int64_t remain = d / axis_dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int64_t step = BlockDim * remain;

  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);
}

template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiff(const T* logits_data, T* max_data,
                                           T* softmax, int d, int axis_dim) {
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;
  int step = BlockDim * remain;

  T diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }

  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}
#endif  // @} End HIP Seperate Kernel for RowReductionForDiffMaxSum

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim>
static __global__ void RowReductionForSoftmaxAndCrossEntropy(
    const T* logits_data, const T* labels_data, T* loss_data, T* softmax,
    int64_t d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax, labels data view as [n, axis_dim, remain]
  // loss_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / axis_dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = softmax[beg_idx] - block_log_diff_max_sum;
  softmax[beg_idx] = exp_on_device(tmp);
  auto loss = -labels_data[beg_idx] * tmp;
  int64_t step = BlockDim * remain;
  beg_idx += step;
  while (beg_idx < end_idx) {
    tmp = softmax[beg_idx] - block_log_diff_max_sum;
    softmax[beg_idx] = exp_on_device(tmp);
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += step;
  }

  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim>
static __global__ void RowReductionForCrossEntropy(const T* logits_data,
                                                   const T* labels_data,
                                                   T* loss_data, int d,
                                                   int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax, labels data view as [n, axis_dim, remain]
  // loss_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = log_on_device(logits_data[beg_idx]);  // when not with softmax,
                                                   // softmax is stored in
                                                   // logits_data
  auto loss = -labels_data[beg_idx] * tmp;
  int step = BlockDim * remain;
  beg_idx += step;
  while (beg_idx < end_idx) {
    tmp = log_on_device(logits_data[beg_idx]);  // when not with softmax,
                                                // softmax is stored in
                                                // logits_data
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += step;
  }

  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

template <typename T>
static void SoftmaxWithCrossEntropyFusedKernel(
    const T* logits_data, const T* labels_data, T* softmax_data, T* loss_data,
    int64_t n, int64_t d, int axis_dim, gpuStream_t stream) {
#ifdef __HIPCC__
  constexpr int kMaxBlockDim = 256;
#else
  constexpr int kMaxBlockDim = 512;
#endif
  int64_t block_dim = axis_dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(axis_dim)));
  int64_t grid_dim = n * d / axis_dim;
#ifdef __HIPCC__
#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, d, axis_dim);                                \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSum<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, softmax_data, d, axis_dim);                  \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(RowReductionForSoftmaxAndCrossEntropy<T, BlockDim>),   \
        dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, labels_data,   \
        loss_data, softmax_data, d, axis_dim);                                 \
    break
#else
#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    RowReductionForMax<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(        \
        logits_data, loss_data, d, axis_dim);                                  \
    RowReductionForDiffMaxSum<T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>( \
        logits_data, loss_data, softmax_data, d, axis_dim);                    \
    RowReductionForSoftmaxAndCrossEntropy<                                     \
        T, BlockDim><<<grid_dim, BlockDim, 0, stream>>>(                       \
        logits_data, labels_data, loss_data, softmax_data, d, axis_dim);       \
    break
#endif

  switch (block_dim) {
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }

#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

// not with softmax
template <typename T>
static void CrossEntropyFusedKernel(const T* logits_data, const T* labels_data,
                                    T* loss_data, int n, int d, int axis_dim,
                                    gpuStream_t stream) {
  constexpr int kMaxBlockDim = 512;
  int block_dim = axis_dim >= kMaxBlockDim
                      ? kMaxBlockDim
                      : (1 << static_cast<int>(std::log2(axis_dim)));
  int grid_dim = n * d / axis_dim;

#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                \
  case BlockDim:                                                              \
    RowReductionForCrossEntropy<T,                                            \
                                BlockDim><<<grid_dim, BlockDim, 0, stream>>>( \
        logits_data, labels_data, loss_data, d, axis_dim);                    \
    break

  switch (block_dim) {
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(512);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(256);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(128);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(64);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(32);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(16);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(8);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(4);
    CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(2);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Block Dimension must be 2^n in softmax_with_cross_entropy_op."));
      break;
  }

#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const bool use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      const Tensor* softmax = context.Input<Tensor>("Logits");
      const Tensor* labels = context.Input<Tensor>("Label");
      Tensor* softmax_out = context.Output<Tensor>("Softmax");
      Tensor* loss = context.Output<Tensor>("Loss");

      const int rank = softmax->dims().size();
      const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
      const int axis_dim = softmax->dims()[axis];

      const int n = SizeToAxis(axis, softmax->dims());
      const int d = SizeFromAxis(axis, softmax->dims());

      auto* softmax_out_data = softmax_out->mutable_data<T>(context.GetPlace());
      auto* loss_data = loss->mutable_data<T>(context.GetPlace());

      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      if (axis_dim == 1) {
        set_constant(context.cuda_device_context(), softmax_out,
                     static_cast<T>(1));
        return;
      }

      auto soft_label = context.Attr<bool>("soft_label");
      auto ignore_index = context.Attr<int>("ignore_index");

      Tensor softmax_2d, labels_2d, loss_2d, softmax_out_2d;
      softmax_2d.ShareDataWith(*softmax).Resize({n, d});
      labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
      loss_2d.ShareDataWith(*loss).Resize({n, 1});
      softmax_out_2d.ShareDataWith(*softmax_out).Resize({n, d});

      // math::CrossEntropyFunctor support axis is the last
      if (axis == -1) {
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            soft_label, ignore_index, axis_dim);
        return;
      }

      // if axis is not the last, we need a new impliment
      if (soft_label) {
        auto* logits_data = softmax->data<T>();
        auto* labels_data = labels->data<T>();
        CrossEntropyFusedKernel(logits_data, labels_data, loss_data, n, d,
                                axis_dim,
                                context.cuda_device_context().stream());
      } else {  // HardLabel
        auto* logits_data = softmax->data<T>();
        auto* labels_data = labels->data<int64_t>();
        int threads = 128;
        int blocks = (n * d / axis_dim + threads - 1) / threads;
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          CrossEntropyHardLabel<T, true><<<
              blocks, threads, 0, context.cuda_device_context().stream()>>>(
              loss_data, logits_data, labels_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          CrossEntropyHardLabel<T, false><<<
              blocks, threads, 0, context.cuda_device_context().stream()>>>(
              loss_data, logits_data, labels_data, n, axis_dim, d / axis_dim,
              ignore_index);
        }
      }

      // cause of input is softmax
      // copy to output softmax, directly
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), softmax_out);

      return;
    }

    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    const int64_t n = SizeToAxis(axis, logits->dims());
    const int64_t d = SizeFromAxis(axis, logits->dims());

    auto* softmax_data = softmax->mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->mutable_data<T>(context.GetPlace());

    if (axis_dim == 1) {
      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), softmax, static_cast<T>(1));
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      return;
    }

    auto soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    if (soft_label) {
      auto* logits_data = logits->data<T>();
      auto* labels_data = labels->data<T>();
      SoftmaxWithCrossEntropyFusedKernel(
          logits_data, labels_data, softmax_data, loss_data, n, d, axis_dim,
          context.cuda_device_context().stream());
    } else {
      if (!context.Attr<bool>("numeric_stable_mode")) {
        // CUDNN kernel only suppoer 2-D tensor and perfome softmax on last dim
        Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
        logits_2d.ShareDataWith(*logits).Resize({n, d});
        softmax_2d.ShareDataWith(*softmax).Resize({n, d});
        labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
        loss_2d.ShareDataWith(*loss).Resize({n, 1});
        math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                       &logits_2d, &softmax_2d);
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            false, ignore_index, axis_dim);
      } else {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<int64_t>();
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          SoftmaxWithCrossEntropyHardLabel<T, true>(
              context.cuda_device_context(), rank, axis, logits_data,
              labels_data, loss_data, softmax_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          SoftmaxWithCrossEntropyHardLabel<T, false>(
              context.cuda_device_context(), rank, axis, logits_data,
              labels_data, loss_data, softmax_data, n, axis_dim, d / axis_dim,
              ignore_index);
        }
      }
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    T* logit_grad_data = logit_grad->data<T>();

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int64_t n = SizeToAxis(axis, logit_grad->dims());
    const int64_t d = SizeFromAxis(axis, logit_grad->dims());
    const int64_t remain = d / axis_dim;

    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    auto use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      if (context.Attr<bool>("soft_label")) {
        int grid = (n * d + block - 1) / block;
        const T* label_data = labels->data<T>();
        SoftLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, label_data, n, d, remain);
      } else {
        Tensor logits_grad_2d;
        logits_grad_2d.ShareDataWith(*logit_grad).Resize({n, d});
        int grid = (n * remain + block - 1) / block;
        const int64_t* label_data = labels->data<int64_t>();
        HardLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
            logit_grad_data, label_data, n, d, remain, ignore_index);
        int num = n * d;
        grid = (num + block - 1) / block;
        ScaleCrossEntropyGradient<T><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, num, d, remain, label_data,
            ignore_index);
      }

      return;
    }

    // with softmax, continue

    if (context.Attr<bool>("soft_label")) {
      int64_t grid = (n * d + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      const int64_t* label_data = labels->data<int64_t>();
      int grid = (n * d + block - 1) / block;
      SoftmaxWithCrossEntropyGradHardLabel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d / remain, remain,
          ignore_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>);
#else
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
#endif
