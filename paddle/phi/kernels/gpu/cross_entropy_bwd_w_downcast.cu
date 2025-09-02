/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/cross_entropy_grad_kernel.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

/*
  Vectorized wrapper of softmax with cross entropy grad hard label.
  Optimized with float4 vectorization for memory coalescing and improved
  throughput.
*/
template <typename T, typename LabelT, typename LogitT>
__global__ void SoftmaxWithCrossEntropyGradHardLabelVectorized(
    LogitT* __restrict__ logits_grad,
    const T* __restrict__ loss_grad,
    const T* __restrict__ softmax,
    const LabelT* __restrict__ labels,
    const int64_t n,
    const int64_t dim,
    const int64_t d,
    const int ignore_index) {
  // Vectorized load/store with float4 for 128-bit memory transactions
  constexpr int VEC_SIZE = 4;
  using VecT = typename phi::AlignedVector<LogitT, VEC_SIZE>;
  using SoftmaxVecT = typename phi::AlignedVector<T, VEC_SIZE>;

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t vec_id = tid * VEC_SIZE;

  // Ensure we don't exceed bounds
  if (vec_id >= n * dim * d) return;

  // Compute indices for vectorized access
  int64_t idx_n = vec_id / (d * dim);
  int64_t idx_dim_start = (vec_id / d) % dim;
  int64_t idx_d = vec_id % d;
  int64_t ids = idx_n * d + idx_d;

  // Load label once per thread
  auto lbl = static_cast<int64_t>(labels[ids]);

  if (lbl == ignore_index) {
    // Vectorized zero fill for ignore_index
    VecT* vec_grad = reinterpret_cast<VecT*>(&logits_grad[vec_id]);
    VecT zero_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      zero_vec.val[i] = static_cast<LogitT>(0.0f);
    }
    *vec_grad = zero_vec;
    return;
  }

  // Vectorized load of softmax values
  SoftmaxVecT softmax_vec;
  const SoftmaxVecT* softmax_ptr =
      reinterpret_cast<const SoftmaxVecT*>(&softmax[vec_id]);
  softmax_vec = *softmax_ptr;

  // Load loss gradient (broadcast across vector elements)
  T loss_grad_val = loss_grad[ids];

  // Vectorized computation
  VecT grad_vec;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    int64_t current_dim = idx_dim_start + i;
    if (current_dim < dim) {  // Bounds check for partial vectors
      float softmax_val = static_cast<float>(softmax_vec.val[i]);
      float grad_val;

      if (lbl == current_dim) {
        grad_val = (softmax_val - 1.0f) * static_cast<float>(loss_grad_val);
      } else {
        grad_val = softmax_val * static_cast<float>(loss_grad_val);
      }

      grad_vec.val[i] = static_cast<LogitT>(grad_val);
    } else {
      grad_vec.val[i] = static_cast<LogitT>(0.0f);
    }
  }

  // Vectorized store
  VecT* grad_ptr = reinterpret_cast<VecT*>(&logits_grad[vec_id]);
  *grad_ptr = grad_vec;
}

/*
  Specialized kernel for dimensions not divisible by vector size
  Uses warp-level primitives for better performance on irregular sizes
*/
template <typename T, typename LabelT, typename LogitT>
__global__ void SoftmaxWithCrossEntropyGradHardLabelWarp(
    LogitT* __restrict__ logits_grad,
    const T* __restrict__ loss_grad,
    const T* __restrict__ softmax,
    const LabelT* __restrict__ labels,
    const int64_t n,
    const int64_t dim,
    const int64_t d,
    const int ignore_index) {
  const int warps_per_block = 4;
  const int threads_per_warp = 32;
  const int threads_per_block = warps_per_block * threads_per_warp;

  int tid = blockIdx.x * threads_per_block + threadIdx.x;
  int warp_id = threadIdx.x / threads_per_warp;
  int lane_id = threadIdx.x % threads_per_warp;

  // Process multiple elements per thread using warp-level parallelism
  int64_t elements_per_thread =
      (n * dim * d + gridDim.x * threads_per_block - 1) /
      (gridDim.x * threads_per_block);

  for (int e = 0; e < elements_per_thread; ++e) {
    int64_t idx = tid + e * gridDim.x * threads_per_block;
    if (idx >= n * dim * d) break;

    int64_t idx_n = idx / (d * dim);
    int64_t idx_dim = (idx / d) % dim;
    int64_t idx_d = idx % d;
    int64_t ids = idx_n * d + idx_d;

    auto lbl = static_cast<int64_t>(labels[ids]);

    if (lbl == ignore_index) {
      logits_grad[idx] = static_cast<LogitT>(0.0f);
    } else if (lbl == idx_dim) {
      logits_grad[idx] =
          static_cast<LogitT>((static_cast<float>(softmax[idx]) - 1.0f) *
                              static_cast<float>(loss_grad[ids]));
    } else {
      logits_grad[idx] =
          static_cast<LogitT>(static_cast<float>(softmax[idx]) *
                              static_cast<float>(loss_grad[ids]));
    }
  }
}

/*
  Optimized kernel selector based on problem size and alignment
*/
template <typename T, typename LabelT, typename LogitT>
void LaunchOptimizedCrossEntropyGradKernel(const GPUContext& dev_ctx,
                                           LogitT* logits_grad,
                                           const T* loss_grad,
                                           const T* softmax,
                                           const LabelT* labels,
                                           const int64_t n,
                                           const int64_t dim,
                                           const int64_t d,
                                           const int ignore_index) {
  const int64_t total_elements = n * dim * d;
  auto stream = dev_ctx.stream();

  // Check alignment for vectorized kernel
  bool is_aligned = (reinterpret_cast<uintptr_t>(logits_grad) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(softmax) % 16 == 0) &&
                    (total_elements % 4 == 0);

  if (is_aligned && total_elements >= 1024) {
    // Use vectorized kernel for aligned, large problems
    constexpr int VEC_SIZE = 4;
    const int threads_per_block = 256;
    const int vec_elements = total_elements / VEC_SIZE;
    const int blocks =
        (vec_elements + threads_per_block - 1) / threads_per_block;

    SoftmaxWithCrossEntropyGradHardLabelVectorized<T, LabelT, LogitT>
        <<<blocks, threads_per_block, 0, stream>>>(
            logits_grad, loss_grad, softmax, labels, n, dim, d, ignore_index);
  } else {
    // Use warp-specialized kernel for irregular sizes
    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * 32;
    const int blocks =
        std::min(1024,
                 static_cast<int>((total_elements + threads_per_block - 1) /
                                  threads_per_block));

    SoftmaxWithCrossEntropyGradHardLabelWarp<T, LabelT, LogitT>
        <<<blocks, threads_per_block, 0, stream>>>(
            logits_grad, loss_grad, softmax, labels, n, dim, d, ignore_index);
  }
}

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel(
    const GPUContext& dev_ctx,
    const DenseTensor& label,
    const DenseTensor& softmax,
    const DenseTensor& loss_grad,
    int axis,
    DenseTensor* logits_grad) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      phi::AllocationType::GPU,
      common::errors::Unavailable("softmax_with_cross_entropy operator's "
                                  "CUDA kernel only runs on GPU device."));

  using LogitT = phi::bfloat16;
  const T* loss_grad_data = loss_grad.data<T>();
  DenseTensor* logit_grad = logits_grad;

  LogitT* logit_grad_data = nullptr;
  logit_grad_data = dev_ctx.template Alloc<LogitT>(logit_grad);

  const int rank = logit_grad->dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = logit_grad->dims()[axis_v];

  const int64_t n = phi::funcs::SizeToAxis(axis_v, logit_grad->dims());
  const int64_t d = phi::funcs::SizeFromAxis(axis_v, logit_grad->dims());
  const int64_t remain = d / axis_dim;

  const T* softmax_data = softmax.data<T>();
  const auto* label_data = label.data<LabelT>();

  // Launch optimized kernel with automatic selection
  LaunchOptimizedCrossEntropyGradKernel<T, LabelT, LogitT>(dev_ctx,
                                                           logit_grad_data,
                                                           loss_grad_data,
                                                           softmax_data,
                                                           label_data,
                                                           n,
                                                           axis_dim,
                                                           remain,
                                                           -100);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxBwdWithDowncastKernel(const Context& dev_ctx,
                                                  const DenseTensor& label,
                                                  const DenseTensor& softmax,
                                                  const DenseTensor& loss_grad,
                                                  DenseTensor* logits_grad) {
  constexpr int axis = -1;
  if (logits_grad->numel() == 0) {
    dev_ctx.template Alloc<phi::bfloat16>(logits_grad);
    return;
  }
  auto dtype = label.dtype();
  PD_VISIT_INTEGRAL_TYPES(
      dtype, "CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel", ([&] {
        CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel<T, data_t>(
            dev_ctx, label, softmax, loss_grad, axis, logits_grad);
      }));
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax_bwd_w_downcast,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxBwdWithDowncastKernel,
                   float,
                   double,
                   phi::float16) {}
