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
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

#define ALIGN_BYTES 16

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;
namespace kps = phi::kps;

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

template <typename Tx, typename Ty = Tx>
struct ExpAddFunctor {
  HOSTDEVICE inline ExpAddFunctor(Tx max) : max(max) {}

  HOSTDEVICE inline Ty operator()(const Tx& sum, const Tx& x) const {
    return static_cast<Ty>(sum + std::exp(x - max));
  }

 private:
  Tx max;
};

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
template <typename T, typename LabelT, bool IgnoreIndex>
__global__ void CrossEntropyHardLabel(T* loss, const T* softmax,
                                      const LabelT* labels, const int n,
                                      const int dim, const int d,
                                      const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;

  // thread ids compute loss[ids] using softmax[idx]
  if (ids < n * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    if (lbl < 0) {  // label is negative
      loss[ids] = static_cast<T>(0.0);
    } else {  // label is positive of zero
      int64_t idx = idx_n * dim * d + lbl * d + idx_d;
      if (IgnoreIndex == true) {
        // IgnoreIndex is true
        if (lbl == ignore_idx) {
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
template <typename T, typename LabelT, bool IgnoreIndex>
__global__ void CrossEntropyExpHardLabel(T* loss, T* softmax,
                                         const LabelT* labels, const int n,
                                         const int dim, const int d,
                                         const int ignore_idx) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    if (IgnoreIndex == true) {
      // IgnoreIndex is true
      if (idx_dim == lbl) {
        if (lbl == ignore_idx) {
          loss[ids] = static_cast<T>(0.0);
        } else {
          loss[ids] = -softmax[idx];
        }
      }
    } else {
      // IgnoreIndex is false
      if (lbl >= 0 && lbl < dim) {
        if (lbl == idx_dim) {
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
template <typename T, typename LabelT, typename VecT, typename AccT,
          int Log2Elements, SoftmaxMode mode, bool IgnoreIndex>
__global__ void WarpSoftmaxForward(T* loss, T* softmax, const T* src,
                                   const LabelT* label, const int batch_size,
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
  phi::WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

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
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

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
            auto lbl = static_cast<int64_t>(label[first_batch + i]);
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (lbl == loss_idx) {
                if (lbl != ignore_index) {
                  loss[first_batch + i] = -logsoftmax;
                } else {
                  loss[first_batch + i] = static_cast<T>(0.0);
                }
              }
            } else {
              // IgnoreIndex is false
              if (lbl >= 0 && lbl < element_count) {
                if (lbl == loss_idx) {
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
            auto lbl = static_cast<int64_t>(label[first_batch + i]);
            if (IgnoreIndex == true) {
              // IgnoreIndex is true
              if (lbl == loss_idx && lbl != ignore_index) {
                loss[first_batch + i] = -logsoftmax;
              }
            } else {
              // IgnoreIndex is false
              if (lbl >= 0 && lbl < element_count) {
                if (lbl == loss_idx) {
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

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, LabelT, VecT, AccT)   \
  case Log2Elements:                                                  \
    WarpSoftmaxForward<T, LabelT, VecT, AccT, Log2Elements, mode,     \
                       IgnoreIndex><<<blocks, threads, 0, stream>>>(  \
        loss, softmax, src, label, batch_size, stride, element_count, \
        ignore_index);                                                \
    break;

/*
  Wrapper of softmax with cross entropy forward hard label.
*/
template <typename T, typename LabelT, SoftmaxMode mode, bool IgnoreIndex>
void SwitchWarpSoftmaxForward(T* loss, T* softmax, const T* src,
                              const LabelT* label, const int batch_size,
                              const int stride, const int element_count,
                              const int ignore_index, gpuStream_t stream) {
  using AccT = typename details::MPTypeTrait<T>::Type;

  // use 128 threads per block to maximimize gpu utilization
  const int log2_elements = static_cast<int>(Log2Ceil(element_count));
  const int kDimCeil = 1 << log2_elements;
  int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / kWarpSize);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
  dim3 threads(kWarpSize, warps_per_block, 1);

  switch (log2_elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, LabelT, T, AccT);
    default:
      break;
  }
}

template <typename T, bool IgnoreIndex>
__device__ __forceinline__ void ComputeLoss(T* loss, const T loss_value,
                                            const int label_id,
                                            const int64_t label_value,
                                            const int tid, const int vec_size,
                                            const int offset,
                                            const int ignore_index) {
  int loss_id = vec_size * tid + offset;
  if (IgnoreIndex) {
    if (label_value == loss_id) {
      if (label_value == ignore_index) {
        loss[label_id] = static_cast<T>(0.0f);
      } else {
        loss[label_id] = loss_value;
      }
    }
  } else {
    if (label_value == loss_id) {
      loss[label_id] = loss_value;
    }
  }
}

template <typename T, typename AccT, int VecSize, class ReduceFunctor>
__device__ __forceinline__ AccT ThreadReduce(const T* input, int size,
                                             const int offset, AccT init,
                                             ReduceFunctor reducer) {
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  AccT val = init;

  if (offset > 0) {
    input -= offset;
    size += offset;
    if (tid >= offset) {
      val = reducer(val, input[tid]);
    }
    size -= blockDim.x;
    input += blockDim.x;
  }
  int remain = size % (VecSize * blockDim.x);

  T ins[VecSize];
  VecT* ins_vec = reinterpret_cast<VecT*>(&ins);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    *ins_vec = reinterpret_cast<const VecT*>(input)[tid];

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      val = reducer(val, ins[i]);
    }
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    val = reducer(val, input[tid]);
  }
  return val;
}

template <typename T, typename AccT, typename LabelT, int VecSize,
          bool IgnoreIndex>
__device__ __forceinline__ void VectorizedSoftmaxForwardImpl(
    T* loss, T* softmax, const T* logits, const LabelT* label, int size,
    const int offset, const phi::LogSoftmaxForwardFunctor<AccT>& func,
    const int ignore_index) {
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  int label_id = blockIdx.x;
  auto label_value = static_cast<int64_t>(label[label_id]);
  const bool label_valid = label_value >= 0 && label_value < size;
  int loss_id_offset = 0;

  if (offset > 0) {
    logits -= offset;
    softmax -= offset;
    size += offset;
    loss_id_offset -= offset;
    if (tid >= offset) {
      AccT log_softmax = func(static_cast<AccT>(logits[tid]));
      softmax[tid] = static_cast<T>(std::exp(log_softmax));
      // loss
      if (label_valid) {
        ComputeLoss<T, IgnoreIndex>(loss, static_cast<T>(-log_softmax),
                                    label_id, label_value, tid, 1,
                                    loss_id_offset, ignore_index);
      }
    }
    size -= blockDim.x;
    logits += blockDim.x;
    softmax += blockDim.x;
    loss_id_offset += blockDim.x;
  }
  int remain = size % (VecSize * blockDim.x);

  T ins[VecSize];
  T outs[VecSize];
  VecT* ins_vec = reinterpret_cast<VecT*>(&ins);
  VecT* outs_vec = reinterpret_cast<VecT*>(&outs);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    // read
    *ins_vec = reinterpret_cast<const VecT*>(logits)[tid];

#pragma unroll
    // compute
    for (int i = 0; i < VecSize; ++i) {
      AccT log_softmax = func(static_cast<AccT>(ins[i]));
      outs[i] = static_cast<T>(std::exp(log_softmax));

      // loss
      if (label_valid) {
        ComputeLoss<T, IgnoreIndex>(loss, static_cast<T>(-log_softmax),
                                    label_id, label_value, tid, VecSize,
                                    loss_id_offset + i, ignore_index);
      }
    }

    // write
    reinterpret_cast<VecT*>(softmax)[tid] = *outs_vec;
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    AccT log_softmax = func(static_cast<AccT>(logits[tid]));
    softmax[tid] = static_cast<T>(std::exp(log_softmax));

    // loss
    if (label_valid) {
      ComputeLoss<T, IgnoreIndex>(loss, static_cast<T>(-log_softmax), label_id,
                                  label_value, tid, 1, loss_id_offset,
                                  ignore_index);
    }
  }

  // invalid label, write once
  if (!label_valid && threadIdx.x == 0) {
    loss[label_id] = static_cast<T>(0.0f);
  }
}

template <typename T, typename AccT, typename LabelT, int VecSize,
          bool IgnoreIndex>
__device__ __forceinline__ void ScalarSoftmaxForwardImpl(
    T* loss, T* softmax, const T* logits, const LabelT* label, const int size,
    const phi::LogSoftmaxForwardFunctor<AccT>& func, const int ignore_index) {
  int tid = threadIdx.x;
  int remain = size % (VecSize * blockDim.x);
  int label_id = blockIdx.x;
  auto label_value = static_cast<int64_t>(label[label_id]);
  const bool label_valid = label_value >= 0 && label_value < size;

  // main part
  for (; tid < (size - remain); tid += VecSize * blockDim.x) {
    T ins[VecSize];

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      ins[i] = logits[tid + i * blockDim.x];
    }
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      AccT log_softmax = func(static_cast<AccT>(ins[i]));
      softmax[tid + i * blockDim.x] = static_cast<T>(std::exp(log_softmax));
      // loss
      if (label_valid) {
        ComputeLoss<T, IgnoreIndex>(loss, static_cast<T>(-log_softmax),
                                    label_id, label_value, tid, VecSize, i,
                                    ignore_index);
      }
    }
  }

  // tail part
  for (; tid < size; tid += blockDim.x) {
    AccT log_softmax = func(static_cast<AccT>(logits[tid]));
    softmax[tid] = static_cast<T>(std::exp(log_softmax));
    // loss
    if (label_valid) {
      ComputeLoss<T, IgnoreIndex>(loss, static_cast<T>(-log_softmax), label_id,
                                  label_value, tid, 1, 0, ignore_index);
    }
  }

  // invalid label, write once
  if (!label_valid && threadIdx.x == 0) {
    loss[label_id] = static_cast<T>(0.0f);
  }
}

template <typename T, typename AccT, typename LabelT, int VecSize,
          bool IgnoreIndex>
__global__ void VectorizedSoftmaxForward(T* loss, T* softmax, const T* logits,
                                         const LabelT* label,
                                         const int high_dim, const int mid_dim,
                                         const int ignore_index) {
  using VecT = kps::details::VectorType<T, VecSize>;

  // each block deal with one batch
  logits += blockIdx.x * mid_dim;
  softmax += blockIdx.x * mid_dim;

  const int input_offset = ((uint64_t)logits) % ALIGN_BYTES / sizeof(T);
  const int output_offset = ((uint64_t)softmax) % ALIGN_BYTES / sizeof(T);

  // 1. reduce max
  AccT max = ThreadReduce<T, AccT, VecSize, kps::MaxFunctor<AccT>>(
      logits, mid_dim, input_offset, -std::numeric_limits<AccT>::infinity(),
      kps::MaxFunctor<AccT>());
  max = kps::details::BlockXReduce<AccT, kps::MaxFunctor<AccT>>(
      max, kps::MaxFunctor<AccT>());

  // 2. reduce sum
  AccT sum = ThreadReduce<T, AccT, VecSize, ExpAddFunctor<AccT>>(
      logits, mid_dim, input_offset, static_cast<AccT>(0),
      ExpAddFunctor<AccT>(max));
  sum = kps::details::BlockXReduce<AccT, kps::AddFunctor<AccT>>(
      sum, kps::AddFunctor<AccT>());

  // 3. softmax
  phi::LogSoftmaxForwardFunctor<AccT> func(max, sum);
  if (input_offset == output_offset) {
    VectorizedSoftmaxForwardImpl<T, AccT, LabelT, VecSize, IgnoreIndex>(
        loss, softmax, logits, label, mid_dim, input_offset, func,
        ignore_index);
  } else {
    ScalarSoftmaxForwardImpl<T, AccT, LabelT, VecSize, IgnoreIndex>(
        loss, softmax, logits, label, mid_dim, func, ignore_index);
  }
}

template <typename T, typename LabelT, bool IgnoreIndex>
void LaunchVectorizedSoftmaxForward(T* loss, T* softmax, const T* logits,
                                    const LabelT* label, const int high_dim,
                                    const int mid_dim, const int ignore_index,
                                    gpuStream_t stream) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  constexpr int vec_size = sizeof(float4) / sizeof(T);
  const int max_num_threads = 1024;
  int max_block_size = std::min(mid_dim / vec_size, max_num_threads);
  if (vec_size > 1) {
    max_block_size /= 2;
  }

  int block_size = 1;
  while (block_size < max_block_size) {
    block_size *= 2;
  }
  block_size = std::max(block_size, kps::details::kWarpSize);
  dim3 grids(high_dim);
  dim3 blocks(block_size);
  VectorizedSoftmaxForward<T, AccT, LabelT, vec_size,
                           IgnoreIndex><<<grids, blocks, 0, stream>>>(
      loss, softmax, logits, label, high_dim, mid_dim, ignore_index);
}

/*
  Wrapper of softmax with cross entropy hard label.
  - SwitchWarpSoftmaxForward for small size when axis == -1
  - LaunchVectorizedSoftmaxForward for large size when axis == -1
  - cudnn function for axis != -1
*/
template <typename T, typename LabelT, bool IgnoreIndex>
static void SoftmaxWithCrossEntropyHardLabel(
    const platform::CUDADeviceContext& ctx, int rank, int axis,
    const T* logits_data, const LabelT* labels_data, T* loss_data,
    T* softmax_data, int N, int dim, int D, const int ignore_index) {
  auto stream = ctx.stream();
  constexpr int max_dim = 320;
  if (D == 1) {
    if (dim <= max_dim) {  // small size
      const SoftmaxMode mode = SoftmaxMode::kCrossEntropy;
      SwitchWarpSoftmaxForward<T, LabelT, mode, IgnoreIndex>(
          loss_data, softmax_data, logits_data, labels_data, N, dim, dim,
          ignore_index, stream);
    } else {  // large size
      LaunchVectorizedSoftmaxForward<T, LabelT, IgnoreIndex>(
          loss_data, softmax_data, logits_data, labels_data, N, dim,
          ignore_index, stream);
    }
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
    CrossEntropyExpHardLabel<T, LabelT,
                             IgnoreIndex><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, labels_data, N, dim, D, ignore_index);
  }
}

/*
  Wrapper of softmax with cross entropy grad hard label.
*/
template <typename T, typename LabelT>
__global__ void SoftmaxWithCrossEntropyGradHardLabel(
    T* logits_grad, const T* loss_grad, const LabelT* labels, const int64_t n,
    const int64_t dim, const int64_t d, const int ignore_index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    if (lbl == ignore_index) {
      logits_grad[idx] = static_cast<T>(0.0);
    } else if (lbl == idx_dim) {
      logits_grad[idx] =
          (logits_grad[idx] - static_cast<T>(1.0)) * loss_grad[ids];
    } else {
      logits_grad[idx] *= loss_grad[ids];
    }
  }
}

/*
  Cross entropy soft label with dynamic size on axis (log2_elements is
  varibale).
  - if the input is softmax，compute loss with softmax
  - if the input is log_softmax, compute loss with log_softmax and update
  softmax
*/
template <typename T, typename VecT, bool InLogMode = false>
__global__ void CrossEntropySoftLabel(T* loss, T* softmaxwrt, const T* softmax,
                                      const T* labels, const int n,
                                      const int dim, const int d,
                                      int log2_elements) {
  const int kDimCeil = 1 << log2_elements;
  const int kVSize = sizeof(VecT) / sizeof(T);

#ifdef __HIPCC__
  const int kThreadPerBlock = 256;
#else
  const int kThreadPerBlock = 512;
#endif
  const int kBatchPerBlock = 1;
  const int kWarpSize = 32;  // (dim < 32) ? dim : 32;
  const int kBatchSize = 1;
  const int kThreadPerBatch = kThreadPerBlock / kBatchPerBlock;
  const int kWarpPerBatch = kThreadPerBatch / kWarpSize;

  const int kIterations = (dim + kThreadPerBatch - 1) / kThreadPerBatch;
  const int kIterationsV = (kIterations >= kVSize) ? (kIterations / kVSize) : 1;

  const int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  T sum[kBatchSize]{static_cast<T>(0.0)};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    int ids = first_batch + i;
    if (ids >= n * d) break;
    int idx_n = ids / d;
    int idx_d = ids % d;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      int idx_dim = it * kThreadPerBatch + threadIdx.x;
      int idx = idx_n * dim * d + idx_dim * d + idx_d;

      if (idx_n < n && idx_dim < dim) {
        VecT softmaxdata;
        if (InLogMode) {
          softmaxdata = reinterpret_cast<VecT*>(&softmaxwrt[idx])[0];
        } else {
          softmaxdata = reinterpret_cast<const VecT*>(&softmax[idx])[0];
        }
        VecT labelsdata = reinterpret_cast<const VecT*>(&labels[idx])[0];
        T* softmaxptr = reinterpret_cast<T*>(&softmaxdata);
        T* labelsptr = reinterpret_cast<T*>(&labelsdata);
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          if (InLogMode) {
            sum[i] -= softmaxptr[s] * labelsptr[s];
            softmaxptr[s] = Exp(softmaxptr[s]);
          } else {
            sum[i] -= Log(softmaxptr[s]) * labelsptr[s];
          }
        }
        if (InLogMode) {
          reinterpret_cast<VecT*>(&softmaxwrt[idx])[0] = softmaxdata;
        }
      }
    }
  }
  phi::WarpReduceSum<T, kBatchSize, kWarpSize>(sum);
  __syncthreads();

  __shared__ T sumshare[kWarpPerBatch][kBatchPerBlock][kBatchSize];
  if (threadIdx.x % kWarpSize == 0) {
#pragma unroll
    for (int i = 0; i < kBatchSize; i++) {
      sumshare[threadIdx.x / kWarpSize][threadIdx.y][i] = sum[i];
    }
  }
  __syncthreads();

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < kBatchSize; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sumshare[0][threadIdx.y][i];
        for (int s = 1; s < kWarpPerBatch; s++) {
          loss[ids] += sumshare[s][threadIdx.y][i];
        }
      }
    }
  }
}

/*
Core function of softmax with cross entropy forward soft label.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: sum of - sum_{j}{ label_{i,j} * (src_{i,j} - maxvalue_{i} -
log(sum[i]))}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoftLabel(T* loss, T* softmax, const T* src,
                                            const T* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count) {
  const bool LogMode = true;

  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT srcdata[kBatchSize][kIterationsV];
  VecT labeldata[kBatchSize][kIterationsV];

  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    const VecT* label_v =
        reinterpret_cast<const VecT*>(&label[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    // read data
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
        labeldata[i][it] = label_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<AccT>::max();
          reinterpret_cast<T*>(&labeldata[i][it])[s] = 0.0;
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    max_value[i] = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = srcptr_v[0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcptr_v[s]) ? valmax : srcptr_v[s];
      }
      max_value[i] = (max_value[i] > static_cast<AccT>(valmax))
                         ? max_value[i]
                         : static_cast<AccT>(valmax);
    }
  }
  phi::WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          sum[i] += std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
        } else {
          srcptr_v[s] = std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
          sum[i] += static_cast<AccT>(srcptr_v[s]);
        }
      }
    }
  }
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // log_softmax and loss
  AccT sumloss[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    if (LogMode) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T* labelvp = reinterpret_cast<T*>(&labeldata[i][it]);
      VecT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          AccT logsoftmax = static_cast<AccT>(srcvp[s]) - max_value[i] - sum[i];
          sumloss[i] -= logsoftmax * static_cast<AccT>(labelvp[s]);
          tmpvp[s] = std::exp(logsoftmax);
        } else {
          tmpvp[s] = static_cast<AccT>(srcvp[s]) / sum[i];
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      }
    }
  }

  // loss
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sumloss);

  for (int i = 0; i < kBatchSize; i++) {
    if (i >= local_batches) break;
    loss[first_batch + i] = sumloss[i];
  }
}

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VecT, AccT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardSoftLabel<T, VecT, AccT,                                 \
                                Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count);         \
    break;

/*
  Wrapper of softmax with cross entropy forward soft label.
*/
template <typename T>
void SwitchWarpSoftmaxForwardSoftLabel(const int blocks, const dim3 threads,
                                       gpuStream_t stream, T* loss, T* softmax,
                                       const T* src, const T* label,
                                       const int batch_size, const int stride,
                                       const int element_count,
                                       const int log2_elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (log2_elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, T, AccT);
    default:
      break;
  }
}

template <typename T>
static void SoftmaxWithCrossEntropySoftLabel(
    const platform::CUDADeviceContext& ctx, const int rank, const int axis,
    const T* logits_data, const T* labels_data, T* softmax_data, T* loss_data,
    int N, int dim, int D) {
#ifdef __HIPCC__
  constexpr int kMaxBlockDim = 256;
#else
  constexpr int kMaxBlockDim = 512;
#endif
  int64_t block_dim = dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(dim)));

  int64_t grid_dim = N * D;
  constexpr int max_dim = 320;

  const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
  const int kDimCeil = 1 << kDimLog2;
  auto stream = ctx.stream();

  if (D == 1 && dim <= max_dim) {
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchWarpSoftmaxForwardSoftLabel<T>(blocks, threads, stream, loss_data,
                                         softmax_data, logits_data, labels_data,
                                         N, dim, dim, kDimLog2);

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

    const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
    const int kDimCeil = 1 << kDimLog2;
#ifdef __HIPCC__
    int kThreadPerBlock = 256;
#else
    int kThreadPerBlock = 512;
#endif

    int kBatchPerBlock = 1;
    int blocks = (N * D + kBatchPerBlock - 1) / kBatchPerBlock;
    dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

    CrossEntropySoftLabel<T, T, true><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, NULL, labels_data, N, dim, D, kDimLog2);
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

template <typename T, typename LabelT>
__global__ void HardLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const LabelT* labels,
                                                    const int n, const int d,
                                                    const int remain,
                                                    const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = static_cast<int>(labels[index]);
    int idx = idx_n * d + tmp * remain + idx_remain;
    if (ignore_index != tmp) {
      logit_grad[idx] = -static_cast<T>(1.) / logit_grad[idx];
    }
  }
}

template <typename T, typename LabelT>
__global__ void ScaleCrossEntropyGradient(T* logit_grad, const T* loss_grad,
                                          const int num, const int d,
                                          const int remain,
                                          const LabelT* labels,
                                          const int ignore_index) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    int idx_lbl = idx_n * remain + idx_remain;
    int k = (index % d) / remain;
    auto lbl = static_cast<int64_t>(labels[idx_lbl]);
    if (lbl == ignore_index || lbl != k) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    RunSoftmaxWithCrossEntropyFunctor<T>(context, *this);
  }

  template <typename LabelT>
  static void Apply(const framework::ExecutionContext& context,
                    const framework::Tensor& labels, const bool soft_label) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const bool use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      const Tensor* softmax = context.Input<Tensor>("Logits");
      Tensor* softmax_out = context.Output<Tensor>("Softmax");
      Tensor* loss = context.Output<Tensor>("Loss");

      const int rank = softmax->dims().size();
      const int axis =
          phi::funcs::CanonicalAxis(context.Attr<int>("axis"), rank);
      const int axis_dim = softmax->dims()[axis];

      const int n = phi::funcs::SizeToAxis(axis, softmax->dims());
      const int d = phi::funcs::SizeFromAxis(axis, softmax->dims());

      auto* softmax_out_data =
          softmax_out->template mutable_data<T>(context.GetPlace());
      auto* loss_data = loss->template mutable_data<T>(context.GetPlace());

      phi::funcs::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      if (axis_dim == 1) {
        set_constant(context.cuda_device_context(), softmax_out,
                     static_cast<T>(1));
        return;
      }

      auto ignore_index = context.Attr<int>("ignore_index");

      Tensor softmax_2d, labels_2d, loss_2d, softmax_out_2d;
      softmax_2d.ShareDataWith(*softmax).Resize({n, d});
      labels_2d.ShareDataWith(labels).Resize({n, labels.numel() / n});
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
        auto* logits_data = softmax->template data<T>();
        auto* labels_data = labels.template data<T>();

        const int kDimLog2 = static_cast<int>(Log2Ceil(axis_dim));
        const int kDimCeil = 1 << kDimLog2;
#ifdef __HIPCC__
        int kThreadPerBlock = 256;
#else
        int kThreadPerBlock = 512;
#endif
        int kBatchPerBlock = 1;
        int blocks = (n * d + kBatchPerBlock - 1) / kBatchPerBlock;
        dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

        CrossEntropySoftLabel<T, T, false><<<
            blocks, threads, 0, context.cuda_device_context().stream()>>>(
            loss_data, NULL, logits_data, labels_data, n, axis_dim,
            d / axis_dim, kDimLog2);
      } else {  // HardLabel
        auto* logits_data = softmax->template data<T>();
        auto* labels_data = labels.template data<LabelT>();
        int threads = 128;
        int blocks = (n * d / axis_dim + threads - 1) / threads;
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          CrossEntropyHardLabel<T, LabelT, true><<<
              blocks, threads, 0, context.cuda_device_context().stream()>>>(
              loss_data, logits_data, labels_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          CrossEntropyHardLabel<T, LabelT, false><<<
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
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    const int rank = logits->dims().size();
    const int axis = phi::funcs::CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logits->dims()[axis];

    const int64_t n = phi::funcs::SizeToAxis(axis, logits->dims());
    const int64_t d = phi::funcs::SizeFromAxis(axis, logits->dims());

    auto* softmax_data = softmax->template mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->template mutable_data<T>(context.GetPlace());

    if (axis_dim == 1) {
      phi::funcs::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), softmax, static_cast<T>(1));
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      return;
    }

    auto ignore_index = context.Attr<int>("ignore_index");

    if (soft_label) {
      auto* logits_data = logits->template data<T>();
      auto* labels_data = labels.template data<T>();
      SoftmaxWithCrossEntropySoftLabel<T>(
          context.cuda_device_context(), rank, axis, logits_data, labels_data,
          softmax_data, loss_data, n, axis_dim, d / axis_dim);
    } else {
      if (!context.Attr<bool>("numeric_stable_mode")) {
        // CUDNN kernel only suppoer 2-D tensor and perfome softmax on last dim
        Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
        logits_2d.ShareDataWith(*logits).Resize({n, d});
        softmax_2d.ShareDataWith(*softmax).Resize({n, d});
        labels_2d.ShareDataWith(labels).Resize({n, labels.numel() / n});
        loss_2d.ShareDataWith(*loss).Resize({n, 1});
        math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                       &logits_2d, &softmax_2d);
        math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
            context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
            false, ignore_index, axis_dim);
      } else {
        auto* logits_data = logits->template data<T>();
        auto* labels_data = labels.template data<LabelT>();
        if (ignore_index >= 0 && ignore_index < axis_dim) {
          SoftmaxWithCrossEntropyHardLabel<T, LabelT, true>(
              context.cuda_device_context(), rank, axis, logits_data,
              labels_data, loss_data, softmax_data, n, axis_dim, d / axis_dim,
              ignore_index);
        } else {
          SoftmaxWithCrossEntropyHardLabel<T, LabelT, false>(
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
    RunSoftmaxWithCrossEntropyFunctor<T>(context, *this);
  }

  template <typename LabelT>
  static void Apply(const framework::ExecutionContext& context,
                    const framework::Tensor& labels, const bool soft_label) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))
            ->template data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    T* logit_grad_data = logit_grad->template data<T>();

    const int rank = logit_grad->dims().size();
    const int axis = phi::funcs::CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = logit_grad->dims()[axis];

    const int64_t n = phi::funcs::SizeToAxis(axis, logit_grad->dims());
    const int64_t d = phi::funcs::SizeFromAxis(axis, logit_grad->dims());
    const int64_t remain = d / axis_dim;

#ifdef __HIPCC__
    int block = 256;
#else
    int block = 512;
#endif
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    auto use_softmax = context.Attr<bool>("use_softmax");

    // do not with softmax op, and input is softmax
    if (!use_softmax) {
      if (soft_label) {
        int grid = (n * d + block - 1) / block;
        const T* label_data = labels.template data<T>();
        SoftLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, label_data, n, d, remain);
      } else {
        Tensor logits_grad_2d;
        logits_grad_2d.ShareDataWith(*logit_grad).Resize({n, d});
        int grid = (n * remain + block - 1) / block;
        const auto* label_data = labels.template data<LabelT>();
        HardLabelCrossEntropyGradientKernel<T,
                                            LabelT><<<grid, block, 0, stream>>>(
            logit_grad_data, label_data, n, d, remain, ignore_index);
        int num = n * d;
        grid = (num + block - 1) / block;
        ScaleCrossEntropyGradient<T, LabelT><<<grid, block, 0, stream>>>(
            logit_grad_data, loss_grad_data, num, d, remain, label_data,
            ignore_index);
      }

      return;
    }

    // with softmax, continue

    if (soft_label) {
      int64_t grid = (n * d + block - 1) / block;
      const T* label_data = labels.template data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      const auto* label_data = labels.template data<LabelT>();
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
