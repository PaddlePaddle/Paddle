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
static T sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// One norm
template <typename T>
struct odist_calc {
  static inline T backward(const T diff,
                           const T grad,
                           const T dist,
                           const T p) {
    return grad * sign(diff);
  }
};

// less than two norm
template <typename T>
struct lttdist_calc {
  static inline T backward(const T diff,
                           const T grad,
                           const T dist,
                           const T p) {
    T result = (dist == 0.0) ? T(0)
                             : sign(diff) * pow(abs(diff), p - 1.0) * grad /
                                   pow(dist, p - 1.0);
    result = (diff == 0.0) && (p < 1.0) ? static_cast<T>(0.0) : result;
    return result;
  }
};

// Two norm
template <typename T>
struct tdist_calc {
  static inline T backward(const T diff,
                           const T grad,
                           const T dist,
                           const T p) {
    T result = (dist == 0.0) ? static_cast<T>(0) : grad * diff / dist;
    return result;
  }
};

// p norm
template <typename T>
struct pdist_calc {
  static inline T backward(const T diff,
                           const T grad,
                           const T dist,
                           const T p) {
    T result = (dist == 0.0)
                   ? static_cast<T>(0)
                   : diff * pow(abs(diff), p - 2.0) * grad / pow(dist, p - 1.0);
    return result;
  }
};

// inf norm
template <typename T>
struct idist_calc {
  static inline T backward(const T diff,
                           const T grad,
                           const T dist,
                           const T p) {
    T result = grad * sign(diff) *
               (1.0 - std::min(static_cast<T>(1.0),
                               std::ceil(std::abs(std::abs(diff) - dist))));
    return result;
  }
};

namespace phi {

template <typename T, typename F>
inline static void backward_down_column_cdist(const T* t1,
                                              const T* t2,
                                              T* res,
                                              const T* grad_k,
                                              const T* dist_k,
                                              const T p,
                                              const int64_t r1,
                                              const int64_t r2,
                                              const int64_t m,
                                              const int64_t d,
                                              const int64_t gs,
                                              const int64_t l1_size,
                                              const int64_t l2_size) {
  const T* t1_end = t1 + l1_size;
  const T* t2_end = t2 + l2_size;

  for (int64_t l = 0; l < d; ++l) {
    for (; t1 != t1_end; t1 += m, res += m) {
      T res_tmp = *res;
      for (const T* t2_curr = t2; t2_curr != t2_end;
           t2_curr += m, grad_k += gs, dist_k += 1) {
        res_tmp += F::backward(*t1 - *t2_curr, *grad_k, *dist_k, p);
      }

      *res = res_tmp;
    }

    t1_end += l1_size;
    t2_end += l2_size;
    t2 += l2_size;
  }
}

template <typename T, typename Context, typename F>
static void run_backward_parallel_cdist(const DenseTensor& x,
                                        const DenseTensor& y,
                                        const DenseTensor& dist,
                                        const DenseTensor& grad,
                                        const T p,
                                        DenseTensor* result) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> res_dims = phi::vectorize(result->dims());
  const int64_t r1 = x_dims[x_dims.size() - 2];
  const int64_t r2 = y_dims[y_dims.size() - 2];
  const int64_t m = x_dims[x_dims.size() - 1];
  const int64_t d = res_dims[0];
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;

  const int64_t gs = 1;

  const T* const grad_start = grad.data<T>();
  const T* const dist_start = dist.data<T>();
  const T* const t1_start = x.data<T>();
  const T* const t2_start = y.data<T>();
  T* const res_start = result->data<T>();

  int64_t l = 0;
  int64_t end = m;

  const T* i = t1_start + l;
  const T* j = t2_start + l;
  T* res_l = res_start + l;

  for (const T* const res_end = res_start + end; res_l != res_end;
       i += 1, j += 1, res_l += 1) {
    backward_down_column_cdist<T, F>(i,
                                     j,
                                     res_l,
                                     grad_start,
                                     dist_start,
                                     p,
                                     r1,
                                     r2,
                                     m,
                                     d,
                                     gs,
                                     l1_size,
                                     l2_size);
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
  phi::FullLikeKernel<T>(
      dev_ctx, *x_grad, static_cast<T>(0), x_grad->dtype(), x_grad);
  if (p == 0.0) {
  } else if (p == 1.0) {
    run_backward_parallel_cdist<T, Context, odist_calc<T>>(
        x, y, out, out_grad, p, x_grad);
  } else if (p < 2.0) {
    run_backward_parallel_cdist<T, Context, lttdist_calc<T>>(
        x, y, out, out_grad, p, x_grad);
  } else if (p == 2.0) {
    run_backward_parallel_cdist<T, Context, tdist_calc<T>>(
        x, y, out, out_grad, p, x_grad);
  } else if (p == INFINITY) {
    run_backward_parallel_cdist<T, Context, idist_calc<T>>(
        x, y, out, out_grad, p, x_grad);
  } else {
    run_backward_parallel_cdist<T, Context, pdist_calc<T>>(
        x, y, out, out_grad, p, x_grad);
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

  T* x_grad_data = x_grad->data<T>();
  T* y_grad_data = y_grad->data<T>();

  *x_grad = phi::Reshape<T, Context>(dev_ctx, *x_grad, x_tensor_expand_dims);
  *y_grad = phi::Reshape<T, Context>(dev_ctx, *y_grad, y_tensor_expand_dims);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    cdist_grad, CPU, ALL_LAYOUT, phi::CdistGradKernel, float, double) {}
