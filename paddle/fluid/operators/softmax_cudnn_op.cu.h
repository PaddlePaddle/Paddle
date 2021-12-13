/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

// Vectorization trait 4 * sizeof(T)
template <typename T>
class VecT4 {};
template <>
class VecT4<double> {
 public:
  using Type = long4;
};
template <>
class VecT4<float> {
 public:
  using Type = int4;
};
template <>
class VecT4<platform::float16> {
 public:
  using Type = int2;
};

// Vectorization trait 2 * sizeof(T)
template <typename T>
class VecT2 {};
template <>
class VecT2<double> {
 public:
  using Type = int4;
};
template <>
class VecT2<float> {
 public:
  using Type = int2;
};
template <>
class VecT2<platform::float16> {
 public:
  using Type = int;
};

static inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

namespace kps = paddle::operators::kernel_primitives;

template <typename Tx, typename Ty = Tx>
struct ReduceMaxFunctor {
  inline Ty initial() { return -std::numeric_limits<Ty>::infinity(); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return max(a, b);
  }
};

template <typename Tx, typename Ty = Tx>
struct ExpSubFunctor {
  HOSTDEVICE inline ExpSubFunctor() { y = static_cast<Tx>(0.0f); }

  HOSTDEVICE explicit inline ExpSubFunctor(Tx y) : y((Tx)(y)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::exp(x - y));
  }

 private:
  Tx y;
};

template <typename Tx, typename Ty = Tx>
struct ExpMulFunctor {
  HOSTDEVICE inline ExpMulFunctor() { y = static_cast<Tx>(1.0f); }

  HOSTDEVICE explicit inline ExpMulFunctor(Tx y) : y((Tx)(y)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::exp(x) * y);
  }

 private:
  Tx y;
};

template <typename Tx, typename Ty = Tx>
struct UnarySubFunctor {
  HOSTDEVICE inline UnarySubFunctor() { y = static_cast<Tx>(0.0f); }

  HOSTDEVICE explicit inline UnarySubFunctor(Tx y) : y((Tx)(y)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x - y);
  }

 private:
  Tx y;
};

template <typename Tx, typename Ty = Tx>
struct UnaryLogFunctor {
  HOSTDEVICE inline UnaryLogFunctor() {}

  HOSTDEVICE explicit inline UnaryLogFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::log(x));
  }
};

template <typename Tx, typename Ty>
struct DataTransFunctor {
  HOSTDEVICE inline DataTransFunctor() {}

  HOSTDEVICE explicit inline DataTransFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return x == -std::numeric_limits<Tx>::infinity()
               ? -std::numeric_limits<Ty>::infinity()
               : static_cast<Ty>(x);
  }
};

template <typename Tx, typename Ty = Tx>
struct UnaryDivFunctor {
  HOSTDEVICE inline UnaryDivFunctor() { n_inv = static_cast<Tx>(1.0f); }

  HOSTDEVICE explicit inline UnaryDivFunctor(Tx n) : n_inv((Tx)(1.0 / n)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x * n_inv);
  }

 private:
  Tx n_inv;
};

/*
Core function of computing softmax forward for axis=-1.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: (a_{i,j} - maxvalue_{i}) / s_{i}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxForward(T* softmax, const T* src,
                                   const int batch_size, const int stride,
                                   const int element_count) {
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kLoops = kDimCeil / kWarpSize;
  constexpr int kLoopsV = (kLoops >= kVSize) ? (kLoops / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 32) ? 2 : 1;
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  constexpr int kStep = kBatchSize * kLoopsV * kVSize;
  constexpr int kVItem = kLoopsV * kVSize;
  constexpr AccT kLowInf = -std::numeric_limits<AccT>::infinity();
  using kMode = kps::details::ReduceMode;

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // data src
  AccT srcdata[kBatchSize][kLoopsV][kVSize];
  T src_tmp[kBatchSize][kLoopsV][kVSize];
  kps::Init<AccT, kStep>(&srcdata[0][0][0], kLowInf);
  kps::Init<T, kStep>(&src_tmp[0][0][0], -std::numeric_limits<T>::infinity());

  // data dst
  T out_tmp[kBatchSize][kLoopsV][kVSize];

  // max value
  AccT max[kBatchSize];
  kps::Init<AccT, kBatchSize>(&max[0], kLowInf);

  // sum value
  AccT sum[kBatchSize] = {0};

// read data from global memory
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&src_tmp[i][0][0]);
    kps::ReadData<VecT, VecT, kLoopsV, 1, 1, true>(
        &reg_v[0], &src_v[0], idx_max_v[i], 0, kWarpSize, 1);
    kps::ElementwiseUnary<T, AccT, kVItem, 1, 1, DataTransFunctor<T, AccT>>(
        &srcdata[i][0][0], &src_tmp[i][0][0], DataTransFunctor<T, AccT>());
  }

  // compute max
  kps::Reduce<AccT, kVItem, kBatchSize, 1, ReduceMaxFunctor<AccT>,
              kMode::kLocalMode>(&max[0], &srcdata[0][0][0],
                                 ReduceMaxFunctor<AccT>(), true);
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max);

  // compute sum
  for (int i = 0; i < kBatchSize; ++i) {
    kps::ElementwiseUnary<AccT, AccT, kVItem, 1, 1, ExpSubFunctor<AccT>>(
        &srcdata[i][0][0], &srcdata[i][0][0], ExpSubFunctor<AccT>(max[i]));
  }
  kps::Reduce<AccT, kVItem, kBatchSize, 1, kps::AddFunctor<AccT>,
              kMode::kLocalMode>(&sum[0], &srcdata[0][0][0],
                                 kps::AddFunctor<AccT>(), true);
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write data to global memory
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&out_tmp[i][0][0]);
    kps::ElementwiseUnary<AccT, T, kVItem, 1, 1, UnaryDivFunctor<AccT>>(
        &out_tmp[i][0][0], &srcdata[i][0][0], UnaryDivFunctor<AccT>(sum[i]));
    kps::WriteData<VecT, VecT, kLoopsV, 1, 1, true>(
        &softmax_v[0], &reg_v[0], idx_max_v[i], 0, kWarpSize, 1);
  }
}

/*
Core function of computing softmax backward for axis=-1.
The computation includes
  - Compute sum of exp batch: s_{i} = sum_{j} {src_{i,j} * grad_{i,j}
  - Compute src_{i,j} * ( grad_{i,j}) - s_{i} )
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxBackward(T* dst, const T* grad, const T* src,
                                    int batch_size, int stride,
                                    int element_count) {
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kLoops = kDimCeil / kWarpSize;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;
  constexpr int kLoopsV = (kLoops >= kVSize) ? (kLoops / kVSize) : 1;
  int element_count_v = element_count / kVSize;
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = min(batch_size - first_batch, kBatchSize);

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // read data from global memory
  VecT src_reg[kBatchSize][kLoopsV];
  VecT grad_reg[kBatchSize][kLoopsV];
  VecT k_value;
  for (int s = 0; s < kVSize; s++) {
    reinterpret_cast<T*>(&k_value)[s] = 0.0;
  }
  kps::Init<VecT, kBatchSize * kLoopsV>(&src_reg[0][0], k_value);
  kps::Init<VecT, kBatchSize * kLoopsV>(&grad_reg[0][0], k_value);
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    int flag = i < local_batches ? 1 : 0;
    int ptr = (first_batch + i) * stride;
    const VecT* src_v = reinterpret_cast<const VecT*>(&src[ptr]);
    const VecT* grad_v = reinterpret_cast<const VecT*>(&grad[ptr]);
    kps::ReadData<VecT, VecT, kLoopsV, 1, 1, true>(
        &src_reg[i][0], &src_v[0], idx_max_v[i], 0, kWarpSize, flag);
    kps::ReadData<VecT, VecT, kLoopsV, 1, 1, true>(
        &grad_reg[i][0], &grad_v[0], idx_max_v[i], 0, kWarpSize, flag);
  }

  // change T to AccT
  AccT src_tmp[kBatchSize][kLoopsV][kVSize];
  AccT grad_tmp[kBatchSize][kLoopsV][kVSize];
  const T* src_ptr = reinterpret_cast<const T*>(&src_reg[0][0]);
  const T* grad_ptr = reinterpret_cast<const T*>(&grad_reg[0][0]);
  constexpr int kStep = kBatchSize * kLoopsV * kVSize;
  constexpr int kVItem = kLoopsV * kVSize;
  kps::ElementwiseUnary<T, AccT, kStep, 1, 1, DataTransFunctor<T, AccT>>(
      &src_tmp[0][0][0], &src_ptr[0], DataTransFunctor<T, AccT>());
  kps::ElementwiseUnary<T, AccT, kStep, 1, 1, DataTransFunctor<T, AccT>>(
      &grad_tmp[0][0][0], &grad_ptr[0], DataTransFunctor<T, AccT>());

  // compute sum
  AccT sum[kBatchSize]{0.0};
  AccT sum_tmp[kBatchSize][kLoopsV][kVSize];
  AccT* gradptr = reinterpret_cast<AccT*>(&grad_tmp[0][0][0]);
  AccT* srcptr = reinterpret_cast<AccT*>(&src_tmp[0][0][0]);
  kps::ElementwiseBinary<AccT, AccT, kStep, 1, 1, kps::MulFunctor<AccT>>(
      &sum_tmp[0][0][0], &gradptr[0], &srcptr[0], kps::MulFunctor<AccT>());
  kps::Reduce<AccT, kVItem, kBatchSize, 1, kps::AddFunctor<AccT>,
              kps::details::ReduceMode::kLocalMode>(
      &sum[0], &sum_tmp[0][0][0], kps::AddFunctor<AccT>(), true);
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // write result to global memory
  AccT out[kBatchSize][kLoopsV][kVSize];
  T out_tmp[kBatchSize][kLoopsV][kVSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;
    AccT* gradptr = reinterpret_cast<AccT*>(&grad_tmp[i][0][0]);
    AccT* srcptr = reinterpret_cast<AccT*>(&src_tmp[i][0][0]);
    kps::ElementwiseUnary<AccT, AccT, kVItem, 1, 1, UnarySubFunctor<AccT>>(
        &out[i][0][0], &gradptr[0], UnarySubFunctor<AccT>(sum[i]));
    kps::ElementwiseBinary<AccT, T, kVItem, 1, 1, kps::MulFunctor<AccT>>(
        &out_tmp[i][0][0], &srcptr[0], &out[i][0][0], kps::MulFunctor<AccT>());
    VecT* dst_v = reinterpret_cast<VecT*>(&dst[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&out_tmp[i][0][0]);
    kps::WriteData<VecT, VecT, kLoopsV, 1, 1, true>(
        &dst_v[0], &reg_v[0], idx_max_v[i], 0, kWarpSize, 1);
  }
}

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, AccT)                      \
  case Log2Elements:                                                       \
    WarpSoftmaxForward<T, VecT, AccT, Log2Elements,                        \
                       LogMode><<<blocks, threads, 0, dev_ctx.stream()>>>( \
        dst, src, batch_size, stride, element_count);                      \
    break;

/*
  Wrapper of softmax formward with template instantiation on size of input.
*/
template <typename T, typename VecT, bool LogMode>
void SwitchWarpSoftmaxForward(const int blocks, const dim3 threads,
                              const platform::CUDADeviceContext& dev_ctx,
                              T* dst, const T* src, const int batch_size,
                              const int stride, const int element_count,
                              int Log2Elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, AccT);
    default:
      break;
  }
}

#define SOFTMAX_WARP_BACKWARD_CASE(Log2Elements, AccT)                      \
  case Log2Elements:                                                        \
    WarpSoftmaxBackward<T, VecT, AccT, Log2Elements,                        \
                        LogMode><<<blocks, threads, 0, dev_ctx.stream()>>>( \
        dst, grad, src, batch_size, stride, element_count);                 \
    break;

/*
Wrapper of softmax backward with template instantiation on size of input.
*/
template <typename T, typename VecT, bool LogMode>
void SwitchWarpSoftmaxBackward(const int blocks, const dim3 threads,
                               const platform::CUDADeviceContext& dev_ctx,
                               T* dst, const T* grad, const T* src,
                               const int batch_size, const int stride,
                               const int element_count, int Log2Elements) {
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_BACKWARD_CASE(0, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(1, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(2, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(3, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(4, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(5, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(6, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(7, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(8, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(9, AccT);
    default:
      break;
  }
}

#undef SOFTMAX_WARP_FORWARD_CASE
#undef SOFTMAX_WARP_BACKWARD_CASE

template <typename T, bool LogMode = false>
void SoftmaxForwardCUDAKernelDriver(const platform::CUDADeviceContext& dev_ctx,
                                    const Tensor& x, const int input_axis,
                                    Tensor* out) {
  auto* out_data = out->data<T>();

  auto dims = x.dims();
  const int rank = dims.size();
  const int axis = CanonicalAxis(input_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int max_dim = 512;
  constexpr int warps_per_block = 4;

  if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
    const int kDimLog2 = static_cast<int>(log2_ceil(dim));
    const int kDimCeil = 1 << kDimLog2;
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 32) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    // vectorization read/write
    using T4 = typename VecT4<T>::Type;
    using T2 = typename VecT2<T>::Type;

    if (dim % 4 == 0) {
      SwitchWarpSoftmaxForward<T, T4, LogMode>(blocks, threads, dev_ctx,
                                               out_data, x.data<T>(), N, dim,
                                               dim, kDimLog2);
    } else if (dim % 2 == 0) {
      SwitchWarpSoftmaxForward<T, T2, LogMode>(blocks, threads, dev_ctx,
                                               out_data, x.data<T>(), N, dim,
                                               dim, kDimLog2);
    } else {
      SwitchWarpSoftmaxForward<T, T, LogMode>(blocks, threads, dev_ctx,
                                              out_data, x.data<T>(), N, dim,
                                              dim, kDimLog2);
    }
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    if (LogMode) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
          handle, platform::CudnnDataType<T>::kOne(), desc_, x.data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data,
          MIOPEN_SOFTMAX_LOG, mode));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSoftmaxForward_V2(
          handle, platform::CudnnDataType<T>::kOne(), desc_, x.data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data,
          MIOPEN_SOFTMAX_ACCURATE, mode));
    }
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    if (LogMode) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
          desc_, x.data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          out_data));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, x.data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
    }
#endif
  }
}

template <typename T, bool LogMode = false>
void SoftmaxBackwardCUDAKernelDriver(const platform::CUDADeviceContext& dev_ctx,
                                     const Tensor& out, const Tensor& dout,
                                     const int input_axis, Tensor* dx) {
  auto* dx_data = dx->data<T>();

  auto dims = out.dims();
  const int rank = dims.size();
  const int axis = CanonicalAxis(input_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int max_dim = 512;
  constexpr int warps_per_block = 4;

  if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
    const int kDimLog2 = log2_ceil(dim);
    const int kDimCeil = 1 << kDimLog2;
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    // vectorization read/write
    using T4 = typename VecT4<T>::Type;
    using T2 = typename VecT2<T>::Type;
    if (dim % 4 == 0) {
      SwitchWarpSoftmaxBackward<T, T4, LogMode>(
          blocks, threads, dev_ctx, dx_data, dout.data<T>(), out.data<T>(), N,
          dim, dim, kDimLog2);
    } else if (dim % 2 == 0) {
      SwitchWarpSoftmaxBackward<T, T2, LogMode>(
          blocks, threads, dev_ctx, dx_data, dout.data<T>(), out.data<T>(), N,
          dim, dim, kDimLog2);
    } else {
      SwitchWarpSoftmaxBackward<T, T, LogMode>(
          blocks, threads, dev_ctx, dx_data, dout.data<T>(), out.data<T>(), N,
          dim, dim, kDimLog2);
    }
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    if (LogMode) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSoftmaxBackward_V2(
          handle, platform::CudnnDataType<T>::kOne(), desc_, out.data<T>(),
          desc_, dout.data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data, MIOPEN_SOFTMAX_LOG, mode));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSoftmaxBackward_V2(
          handle, platform::CudnnDataType<T>::kOne(), desc_, out.data<T>(),
          desc_, dout.data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data, MIOPEN_SOFTMAX_ACCURATE, mode));
    }
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    if (LogMode) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
          desc_, out.data<T>(), desc_, dout.data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, dx_data));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, out.data<T>(), desc_,
          dout.data<T>(), platform::CudnnDataType<T>::kZero(), desc_, dx_data));
    }
#endif
  }
}

}  // namespace operators
}  // namespace paddle
