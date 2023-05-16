//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/cdist_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/dist_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

template <typename T, typename ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

struct Block1D {
  static __forceinline__ __device__ int Tid() { return threadIdx.x; }
  static __forceinline__ __device__ int Warps() {
    return blockDim.x / WARP_SIZE;
  }
};

template <typename T, typename ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % WARP_SIZE;
  const int wid = tid / WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

template <typename T>
static __forceinline__ __device__ T sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// Zero norm
template <typename T>
struct zdist_calc {
  static __forceinline__ __device__ void inc(T* agg, const T diff, const T p) {
    *agg += diff != 0.0;
  }
  static __forceinline__ __device__ T finish(const T agg, const T p) {
    return agg;
  }
  static __forceinline__ __device__ void agg(T* update, const T other) {
    *update += other;
  }
};

// One norm
template <typename T>
struct odist_calc {
  static __forceinline__ __device__ void inc(T* agg, const T diff, const T p) {
    *agg += diff;
  }
  static __forceinline__ __device__ T finish(const T agg, const T p) {
    return agg;
  }
  static __forceinline__ __device__ void agg(T* update, const T other) {
    *update += other;
  }
};

// Two norm
template <typename T>
struct tdist_calc {
  static __forceinline__ __device__ void inc(T* agg, const T diff, const T p) {
    *agg += diff * diff;
  }
  static __forceinline__ __device__ T finish(const T agg, const T p) {
    return sqrt(agg);
  }
  static __forceinline__ __device__ void agg(T* update, const T other) {
    *update += other;
  }
};

// P norm
template <typename T>
struct pdist_calc {
  static __forceinline__ __device__ void inc(T* agg, const T diff, const T p) {
    *agg += pow(diff, p);
  }
  static __forceinline__ __device__ T finish(const T agg, const T p) {
    return pow(agg, static_cast<T>(1) / p);
  }
  static __forceinline__ __device__ void agg(T* update, const T other) {
    *update += other;
  }
};

// Inf norm
template <typename T>
struct idist_calc {
  static __forceinline__ __device__ void inc(T* agg, const T diff, const T p) {
    if (diff > *agg) {
      *agg = diff;
    }
  }
  static __forceinline__ __device__ T finish(const T agg, const T p) {
    return agg;
  }
  static __forceinline__ __device__ void agg(T* update, const T other) {
    if (other > *update) {
      *update = other;
    }
  }
};

namespace phi {

template <typename T, typename F>
struct DistReduceOp {
  __forceinline__ __device__ T combine(T a, T b) const {
    F::agg(&a, b);
    return a;
  }

  __forceinline__ __device__ T warp_shfl_down(T val, int offset) const {
#if defined(PADDLE_WITH_CUDA) && (__CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000)
    T warp_val = __shfl_down_sync(0xffffffff, val, offset, warpSize);
#else
    T warp_val = __shfl_down(val, offset, warpSize);
#endif
    return warp_val;
  }
};

template <typename F, typename T>
__global__ static void cdist_kernel_cuda_impl(T* result,
                                              const T* x,
                                              const T* y,
                                              const T p,
                                              const int64_t y_r,
                                              const int64_t m,
                                              const int64_t r_size,
                                              const int64_t x_l_size,
                                              const int64_t y_l_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / y_r;
  const int64_t j = k % y_r;
  const int stride = blockDim.x;

  const T* const start = x + l * x_l_size + i * m;
  const T* const end = start + m;
  const T* a = start + threadIdx.x;
  const T* b = y + l * y_l_size + j * m + threadIdx.x;

  T agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(&agg, std::abs(*a - *b), p);
  }

  const int kMaxThreadsPerBlock = 256;
  __shared__ T agg_smem[kMaxThreadsPerBlock];
  T agg_init{0.0};
  agg = BlockReduce(agg, DistReduceOp<T, F>{}, agg_init, agg_smem);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = F::finish(agg, p);
  }
}

template <typename T, typename Context>
void cdist_impl(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const float p,
                DenseTensor* out) {
  const int kMaxThreadsPerBlock = 256;
  const auto& x_dims = phi::vectorize(x.dims());
  const auto& y_dims = phi::vectorize(y.dims());
  const int64_t x_r = x_dims[x_dims.size() - 2];
  const int64_t y_r = y_dims[y_dims.size() - 2];
  const int64_t m = x_dims[x_dims.size() - 1];
  const int64_t r_size = x_r * y_r;
  const int64_t x_l_size = x_r * m;
  const int64_t y_l_size = y_r * m;
  const dim3 grid(out->numel());
  const dim3 block(kMaxThreadsPerBlock);

  if (p == 0.0) {
    cdist_kernel_cuda_impl<zdist_calc<T>, T>
        <<<grid, block, 0, dev_ctx.stream()>>>(out->data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               p,
                                               y_r,
                                               m,
                                               r_size,
                                               x_l_size,
                                               y_l_size);
  } else if (p == 1.0) {
    cdist_kernel_cuda_impl<odist_calc<T>, T>
        <<<grid, block, 0, dev_ctx.stream()>>>(out->data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               p,
                                               y_r,
                                               m,
                                               r_size,
                                               x_l_size,
                                               y_l_size);
  } else if (p == 2.0) {
    cdist_kernel_cuda_impl<tdist_calc<T>, T>
        <<<grid, block, 0, dev_ctx.stream()>>>(out->data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               p,
                                               y_r,
                                               m,
                                               r_size,
                                               x_l_size,
                                               y_l_size);
  } else if (p == std::numeric_limits<float>::infinity()) {
    cdist_kernel_cuda_impl<idist_calc<T>, T>
        <<<grid, block, 0, dev_ctx.stream()>>>(out->data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               p,
                                               y_r,
                                               m,
                                               r_size,
                                               x_l_size,
                                               y_l_size);
  } else {
    cdist_kernel_cuda_impl<pdist_calc<T>, T>
        <<<grid, block, 0, dev_ctx.stream()>>>(out->data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               p,
                                               y_r,
                                               m,
                                               r_size,
                                               x_l_size,
                                               y_l_size);
  }
}

template <typename T, typename Context>
void _euclidean_dist(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  DenseTensor x_norm;
  x_norm.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_norm);
  phi::PowKernel<T, Context>(dev_ctx, x, phi::Scalar(2.0), &x_norm);
  x_norm = phi::Sum<T, Context>(
      dev_ctx, x_norm, IntArray({-1}), phi::CppTypeToDataType<T>::Type(), true);
  DenseTensor y_norm;
  y_norm.Resize(y.dims());
  dev_ctx.template Alloc<T>(&y_norm);
  phi::PowKernel<T, Context>(dev_ctx, y, phi::Scalar(2.0), &y_norm);
  y_norm = phi::Sum<T, Context>(
      dev_ctx, y_norm, IntArray({-1}), phi::CppTypeToDataType<T>::Type(), true);

  DenseTensor x_pad;
  x_pad.Resize(x_norm.dims());
  dev_ctx.template Alloc<T>(&x_pad);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, &x_pad, static_cast<T>(1));
  DenseTensor y_pad;
  y_pad.Resize(y_norm.dims());
  dev_ctx.template Alloc<T>(&y_pad);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, &y_pad, static_cast<T>(1));

  DenseTensor x_mul = phi::Scale<T, Context>(dev_ctx, x, -2, 0.0, false);
  DenseTensor x_ =
      phi::Concat<T, Context>(dev_ctx, {&x_mul, &x_norm, &x_pad}, -1);
  DenseTensor y_ = phi::Concat<T, Context>(dev_ctx, {&y, &y_pad, &y_norm}, -1);
  DenseTensor temp = phi::Matmul<T, Context>(dev_ctx, x_, y_, false, true);
  phi::SqrtKernel<T, Context>(dev_ctx, temp, out);
}

template <typename T, typename Context>
void CdistKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 float p,
                 const std::string& compute_mode,
                 DenseTensor* out) {
  // set default value for compute_mode
  int mode = 0;
  if (compute_mode == "use_mm_for_euclid_dist_if_necessary") {
    // use matrix multiplication for euclid distance if necessary
    mode = 0;
  } else if (compute_mode == "use_mm_for_euclid_dist") {
    // use matrix multiplication for euclid distance
    mode = 1;
  } else if (compute_mode == "donot_use_mm_for_euclid_dist") {
    // do not use matrix multiplication for euclid distance
    mode = 2;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unsupported compute_mode: %s",
                                              compute_mode));
  }
  auto x_dims = phi::vectorize(x.dims());
  auto y_dims = phi::vectorize(y.dims());
  auto x_dim = x_dims.size();
  auto y_dim = y_dims.size();

  // Get rows and cols of x and y
  auto x_c = x_dims[x_dim - 1];
  auto y_c = y_dims[y_dim - 1];
  auto x_r = x_dims[x_dim - 2];
  auto y_r = y_dims[y_dim - 2];

  // Get the batch size of x and y
  std::vector<int64_t> x_batch_tensor{};
  std::vector<int64_t> y_batch_tensor{};
  std::copy(
      x_dims.begin(), x_dims.end() - 2, std::back_inserter(x_batch_tensor));
  std::copy(
      y_dims.begin(), y_dims.end() - 2, std::back_inserter(y_batch_tensor));

  // For batch calculation we expand all dimensions (except the last two) to
  // one, with size that equals to product of them. The last two dimensions will
  // stay the same.
  std::vector<int64_t> expand_batch_portion =
      phi::funcs::MatrixGetBroadcastBatchPortion(x_batch_tensor,
                                                 y_batch_tensor);
  std::vector<int64_t> x_tensor_expand_size(expand_batch_portion);
  x_tensor_expand_size.insert(x_tensor_expand_size.end(), {x_r, x_c});
  std::vector<int64_t> y_tensor_expand_size(expand_batch_portion);
  y_tensor_expand_size.insert(y_tensor_expand_size.end(), {y_r, y_c});

  auto expand_batch_product = std::accumulate(expand_batch_portion.begin(),
                                              expand_batch_portion.end(),
                                              1,
                                              std::multiplies<int64_t>());

  std::vector<int64_t> output_shape{std::move(expand_batch_portion)};
  output_shape.insert(output_shape.end(), {x_r, y_r});

  // alloc output memory
  out->Resize(phi::make_ddim(output_shape));
  dev_ctx.template Alloc<T>(out);

  if (x_r == 0 || y_r == 0 || expand_batch_product == 0) {
    phi::EmptyKernel<T>(dev_ctx, output_shape, x.dtype(), out);
    return;
  } else if (x_c == 0) {
    phi::Full<T>(dev_ctx, output_shape, 0, out);
    return;
  }

  std::vector<int64_t> x_tensor_reshaped_dims{expand_batch_product, x_r, x_c};
  std::vector<int64_t> y_tensor_reshaped_dims{expand_batch_product, y_r, y_c};

  IntArray x_tensor_expand_dims(x_tensor_expand_size);
  IntArray y_tensor_expand_dims(y_tensor_expand_size);

  DenseTensor x_tmp = phi::Empty<T>(dev_ctx, x_tensor_expand_dims);
  phi::ExpandKernel<T, Context>(dev_ctx, x, x_tensor_expand_dims, &x_tmp);
  auto x_data = x_tmp.data<T>();

  DenseTensor x_tensor_expanded =
      phi::Reshape<T>(dev_ctx, x_tmp, x_tensor_reshaped_dims);
  x_data = x_tensor_expanded.data<T>();

  DenseTensor y_tmp = phi::Empty<T>(dev_ctx, y_tensor_expand_dims);
  phi::ExpandKernel<T, Context>(dev_ctx, y, y_tensor_expand_dims, &y_tmp);
  DenseTensor y_tensor_expanded =
      phi::Reshape<T>(dev_ctx, y_tmp, y_tensor_reshaped_dims);

  if (p == 2 && (mode == 1 || (mode == 0 && (x_r > 25 || y_r > 25)))) {
    if (expand_batch_product == 1) {
      _euclidean_dist<T>(dev_ctx, x, y, out);
    } else {
      _euclidean_dist<T>(dev_ctx, x_tensor_expanded, y_tensor_expanded, out);
    }
  } else {
    cdist_impl<T>(dev_ctx, x_tensor_expanded, y_tensor_expanded, p, out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(cdist, GPU, ALL_LAYOUT, phi::CdistKernel, float, double) {}
