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
#include "paddle/phi/kernels/dist_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/reshape_kernel.h"

template <typename T>
static T sign(T val) {
  return (T(0) < val) - (val < T(0));
}

// Zero norm
template <typename T>
struct zdist_calc {
  static inline T map(const T diff, const T p) {
    return std::min(std::ceil(std::abs(diff)), static_cast<T>(1.0));
  }
  static inline T red(const T agg, const T up) { return agg + up; }
  static inline T finish(const T agg, const T p) { return agg; }
};

// One norm
template <typename T>
struct odist_calc {
  static inline T map(const T diff, const T p) { return diff; }
  static inline T red(const T agg, const T up) { return agg + up; }
  static inline T finish(const T agg, const T p) { return agg; }
};

// Two norm
template <typename T>
struct tdist_calc {
  static inline T map(const T diff, const T p) { return diff * diff; }
  static inline T red(const T agg, const T up) { return agg + up; }
  static inline T finish(const T agg, const T p) { return std::sqrt(agg); }
};

// P norm
template <typename T>
struct pdist_calc {
  static inline T map(const T diff, const T p) {
    return std::pow(std::abs(diff), p);
  }
  static inline T red(const T agg, const T up) { return agg + up; }
  static inline T finish(const T agg, const T p) {
    return std::pow(agg, 1.0 / p);
  }
};

// Inf norm
template <typename T>
struct idist_calc {
  static inline T map(const T diff, const T p) { return diff; }
  static inline T red(const T agg, const T up) { return std::max(agg, up); }
  static inline T finish(const T agg, const T p) { return agg; }
};

namespace phi {

template <typename F, typename T, typename Context>
static void run_parallel_cdist(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const float p,
                               DenseTensor* result) {
  auto* t1_start = x.data<T>();
  auto* t2_start = y.data<T>();

  auto t1_dims = phi::vectorize(x.dims());
  auto t2_dims = phi::vectorize(y.dims());

  int64_t d = t1_dims[0];
  int64_t x_r = t1_dims[t1_dims.size() - 2];
  int64_t y_r = t2_dims[t2_dims.size() - 2];
  int64_t m = t1_dims[t1_dims.size() - 1];

  auto* res_start = result->data<T>();

  int64_t combs = x_r * y_r;
  int64_t size1 = x_r * m;
  int64_t size2 = y_r * m;

  auto end = combs * d;
  auto* res = res_start;
  auto* res_end = res_start + end;
  auto i = 0;
  auto j = 0;

  auto l = 0;

  while (res != res_end) {
    auto* self_i = t1_start + l * size1 + i;
    auto* self_j = t2_start + l * size2 + j;

    T agg = 0.0;
    for (auto x = 0; x < m; ++x) {
      auto a = *(self_i + x);
      auto b = *(self_j + x);
      agg = F::red(agg, F::map(std::abs(a - b), p));
    }
    *res = F::finish(agg, p);

    res++;
    j += m;
    if (j == size2) {
      j = 0;
      i += m;
      if (i == size1) {
        i = 0;
        l++;
      }
    }
  }
}

template <typename T, typename Context>
void cdist_impl(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const float p,
                DenseTensor* out) {
  if (p == 0.0) {
    run_parallel_cdist<zdist_calc<T>, T, Context>(dev_ctx, x, y, p, out);
  } else if (p == 1.0) {
    run_parallel_cdist<odist_calc<T>, T, Context>(dev_ctx, x, y, p, out);
  } else if (p == 2.0) {
    run_parallel_cdist<tdist_calc<T>, T, Context>(dev_ctx, x, y, p, out);
  } else if (p == INFINITY) {
    run_parallel_cdist<idist_calc<T>, T, Context>(dev_ctx, x, y, p, out);
  } else {
    run_parallel_cdist<pdist_calc<T>, T, Context>(dev_ctx, x, y, p, out);
  }
}

template <typename T, typename Context>
void _euclidean_dist(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  DistKernel<T>(dev_ctx, x, y, 2.0, out);
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
  DenseTensor x_tensor_expanded =
      phi::Reshape<T>(dev_ctx, x_tmp, x_tensor_reshaped_dims);

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

PD_REGISTER_KERNEL(cdist, CPU, ALL_LAYOUT, phi::CdistKernel, float, double) {}
