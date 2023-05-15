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

#include "paddle/phi/kernels/cdist_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

template <typename T>
static __forceinline__ __device__ T sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// One norm
template <typename T>
struct odist_calc {
  static __forceinline__ __device__ T backward(const T diff,
                                               const T grad,
                                               const T dist,
                                               const T p) {
    return grad * sign(diff);
  }
};

// less than two norm
template <typename T>
struct lttdist_calc {
  static __forceinline__ __device__ T backward(const T diff,
                                               const T grad,
                                               const T dist,
                                               const T p) {
    return (dist == 0.0 || diff == 0.0 && p < 1)
               ? 0
               : (sign(diff) * std::abs(diff), p - 1) * grad /
                     std::pow(dist, p - 1);
  }
};

// Two norm
template <typename T>
struct tdist_calc {
  static __forceinline__ __device__ T backward(const T diff,
                                               const T grad,
                                               const T dist,
                                               const T p) {
    return dist == 0.0 ? 0 : grad * diff / dist;
  }
};

// P norm
template <typename T>
struct pdist_calc {
  static __forceinline__ __device__ T backward(const T diff,
                                               const T grad,
                                               const T dist,
                                               const T p) {
    return dist == 0.0 ? 0
                       : grad * pow(std::abs(diff), p - 2.0) * diff /
                             pow(dist, p - 1.0);
  }
};

// Inf norm
template <typename T>
struct idist_calc {
  static __forceinline__ __device__ T backward(const T diff,
                                               const T grad,
                                               const T dist,
                                               const T p) {
    return grad * sign(diff) * (std::abs(diff) == dist);
  }
};

namespace phi {

template <typename T, typename F>
__global__ static void cdist_backward_kernel_impl(T* buffer,
                                                  const T* grad,
                                                  const T* x1,
                                                  const T* x2,
                                                  const T* dist,
                                                  const T p,
                                                  const int64_t r1,
                                                  const int64_t r2,
                                                  const int64_t m,
                                                  const int64_t count,
                                                  const int64_t r_size,
                                                  const int64_t l1_size,
                                                  const int64_t l2_size) {
  const int y =
      (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;

  if (y >= count || init >= m) {
    return;
  }

  const int l = y / r_size;
  const int k = y % r_size;
  const int stride = blockDim.x * gridDim.x;
  const int l_size = r_size * m;

  int64_t i = k / r2;
  int64_t j = k % r2;

  const T grad_k = grad[y];
  const T dist_k = dist[k];

  const T* const start = x1 + l * l1_size + i * m;
  const T* const end = start + m;
  const T* self_i = start + init;
  const T* self_j = x2 + l * l2_size + j * m + init;

  T* buff_i = buffer + l + l_size + (r1 + j + i) * m + init;

  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
    const T res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
  }
}

template <typename T, typename Context>
void cdist_grad_impl(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float p,
                     DenseTensor* x_grad) {
  if (p == 0.0 || out_grad.numel() == 0 || x.numel() == 0 || y.numel() == 0) {
    phi::FullLikeKernel<T>(
        dev_ctx, *x_grad, static_cast<T>(0), x_grad->dtype(), x_grad);
    return;
  }

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> x_grad_dims = phi::vectorize(x_grad->dims());

  const int64_t r1 = x_dims[x_dims.size() - 2];
  const int64_t r2 = y_dims[y_dims.size() - 2];
  const int64_t m = x_dims[x_dims.size() - 1];

  const int64_t batch = x_grad_dims[0];
  const int block_x = 64;
  const int block_y = 16;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);

  const int64_t count = out.numel();
  const int64_t grid_temp = (count + block_y - 1) / block_y;

  const int grid_y = (grid_temp - 1) / 65535 + 1;
  const int grid_z = (grid_temp - 1) / grid_y + 1;

  const dim3 grid(grid_x, grid_y, grid_z);
  const dim3 block(block_x, block_y);

  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;

  DenseTensor buffer = phi::Empty<T>(dev_ctx, {batch, r2, r1, m});
  if (p == 1.0) {
    cdist_backward_kernel_impl<T, odist_calc<T>>
        <<<grid, block, 0, dev_ctx.stream()>>>(buffer.data<T>(),
                                               out_grad.data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               out.data<T>(),
                                               p,
                                               r1,
                                               r2,
                                               m,
                                               count,
                                               r_size,
                                               l1_size,
                                               l2_size);
  } else if (p < 2.0) {
    cdist_backward_kernel_impl<T, lttdist_calc<T>>
        <<<grid, block, 0, dev_ctx.stream()>>>(buffer.data<T>(),
                                               out_grad.data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               out.data<T>(),
                                               p,
                                               r1,
                                               r2,
                                               m,
                                               count,
                                               r_size,
                                               l1_size,
                                               l2_size);
  } else if (p == 2.0) {
    cdist_backward_kernel_impl<T, tdist_calc<T>>
        <<<grid, block, 0, dev_ctx.stream()>>>(buffer.data<T>(),
                                               out_grad.data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               out.data<T>(),
                                               p,
                                               r1,
                                               r2,
                                               m,
                                               count,
                                               r_size,
                                               l1_size,
                                               l2_size);
  } else if (p == std::isinf(p)) {
    cdist_backward_kernel_impl<T, idist_calc<T>>
        <<<grid, block, 0, dev_ctx.stream()>>>(buffer.data<T>(),
                                               out_grad.data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               out.data<T>(),
                                               p,
                                               r1,
                                               r2,
                                               m,
                                               count,
                                               r_size,
                                               l1_size,
                                               l2_size);
  } else {
    cdist_backward_kernel_impl<T, pdist_calc<T>>
        <<<grid, block, 0, dev_ctx.stream()>>>(buffer.data<T>(),
                                               out_grad.data<T>(),
                                               x.data<T>(),
                                               y.data<T>(),
                                               out.data<T>(),
                                               p,
                                               r1,
                                               r2,
                                               m,
                                               count,
                                               r_size,
                                               l1_size,
                                               l2_size);
  }
}

template <typename T, typename Context>
void CdistGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float p,
                     const std::string& compute_mode,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  auto x_dim = x_dims.size();
  auto y_dim = y_dims.size();
  int64_t x_c = x_dims[x_dim - 1];
  int64_t y_c = y_dims[y_dim - 1];
  int64_t x_r = x_dims[x_dim - 2];
  int64_t y_r = y_dims[y_dim - 2];

  std::vector<int64_t> x_batch_tensor(x_dims.begin(), x_dims.end() - 2);
  std::vector<int64_t> y_batch_tensor(y_dims.begin(), y_dims.end() - 2);
  std::vector<int64_t> expand_batch_portion =
      phi::funcs::MatrixGetBroadcastBatchPortion(x_batch_tensor,
                                                 y_batch_tensor);
  std::vector<int64_t> x_tensor_expand_dims(expand_batch_portion);
  x_tensor_expand_dims.insert(x_tensor_expand_dims.end(), {x_r, x_c});
  std::vector<int64_t> y_tensor_expand_dims(expand_batch_portion);
  y_tensor_expand_dims.insert(y_tensor_expand_dims.end(), {y_r, y_c});

  // Compute the linearized batch size of the expanded tensor
  const int64_t batch_product = std::accumulate(expand_batch_portion.begin(),
                                                expand_batch_portion.end(),
                                                1,
                                                std::multiplies<int64_t>());

  if (x_r == 0 || y_r == 0 || x_c == 0 || batch_product == 0) {
    phi::FullLikeKernel<T, Context>(
        dev_ctx, x, static_cast<T>(0), x.dtype(), x_grad);
    phi::FullLikeKernel<T, Context>(
        dev_ctx, y, static_cast<T>(0), y.dtype(), y_grad);
    return;
  }

  DenseTensor x_expanded = x;
  if (x_tensor_expand_dims != x_dims) {
    IntArray x_tensor_expand_size(x_tensor_expand_dims);
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, x_tensor_expand_size, &x_expanded);
  }

  DenseTensor y_expanded = y;
  if (y_tensor_expand_dims != y_dims) {
    IntArray y_tensor_expand_size(y_tensor_expand_dims);
    phi::ExpandKernel<T, Context>(
        dev_ctx, y, y_tensor_expand_size, &y_expanded);
  }

  IntArray x_batched_size({batch_product, x_r, x_c});
  IntArray y_batched_size({batch_product, y_r, y_c});

  DenseTensor out_t = phi::TransposeLast2Dim<T, Context>(dev_ctx, out);
  DenseTensor out_grad_t =
      phi::TransposeLast2Dim<T, Context>(dev_ctx, out_grad);

  *x_grad = phi::Empty<T, Context>(dev_ctx, x_batched_size);
  *y_grad = phi::Empty<T, Context>(dev_ctx, y_batched_size);

  cdist_grad_impl<T, Context>(
      dev_ctx, x_expanded, y_expanded, out, out_grad, p, x_grad);
  cdist_grad_impl<T, Context>(
      dev_ctx, y_expanded, x_expanded, out_t, out_grad_t, p, y_grad);

  *x_grad = phi::Reshape<T, Context>(dev_ctx, *x_grad, x_tensor_expand_dims);
  *y_grad = phi::Reshape<T, Context>(dev_ctx, *y_grad, y_tensor_expand_dims);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    cdist_grad, GPU, ALL_LAYOUT, phi::CdistGradKernel, float, double) {}
