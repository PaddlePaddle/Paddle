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

#pragma once

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/expand_as_kernel.h"
#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/squeeze_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

namespace phi {

using Tensor = DenseTensor;

// check the input other is vector_case or not
static inline bool is_vector_rhs(const DenseTensor& input,
                                 const DenseTensor& other) {
  auto x_dim = input.dims();
  auto y_dim = other.dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();
  std::vector<int64_t> x_dims_vec = phi::vectorize(x_dim);
  std::vector<int64_t> y_dims_vec = phi::vectorize(y_dim);

  std::vector<int64_t>::const_iterator f = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l = x_dims_vec.end() - 1;
  std::vector<int64_t> x_dims_vec_cut(f, l);  // input.shape[:-1]

  std::vector<int64_t> expected_batched_rhs_shape(x_dims_vec_cut);
  bool vector_case =
      y_dim_size == 1 || (x_dim_size - 1 == y_dim_size &&
                          y_dims_vec == (expected_batched_rhs_shape));

  return vector_case;
}

// Prepared for the broadcast operation
static std::vector<int64_t> get_broadcast_batch_portion(
    std::vector<int64_t> x, std::vector<int64_t> y) {
  size_t size_x = x.size();
  size_t size_y = y.size();
  size_t size = std::max(size_x, size_y);
  std::vector<int64_t> batchPortion(size);
  ptrdiff_t i = (ptrdiff_t)size - 1;
  for (; i >= 0; --i) {
    ptrdiff_t offset = size - i - 1;
    ptrdiff_t dim_x = size_x - offset - 1;
    ptrdiff_t dim_y = size_y - offset - 1;
    int64_t x_size = (dim_x >= 0) ? x[dim_x] : 1;
    int64_t y_size = (dim_y >= 0) ? y[dim_y] : 1;
    PADDLE_ENFORCE_EQ(
        (x_size == y_size || x_size == 1 || y_size == 1),
        true,
        phi::errors::PreconditionNotMet(
            "The size of tensor x (%d) must match the size of tensor y "
            "(%d) at non-singleton dimension %d.",
            x_size,
            y_size,
            i));

    batchPortion[i] = x_size != 1 ? x_size : y_size;
  }
  return batchPortion;
}

static inline std::vector<int> convert_to_int_vec(std::vector<int64_t> a) {
  std::vector<int> ret;
  for (size_t i = 0; i < a.size(); i++) {
    ret.emplace_back(static_cast<int>(a[i]));
  }

  return ret;
}

// broadcast the batch dimensions of tensor x and tensor y.
static inline std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_broadcast_dims(const Tensor& x, const Tensor& y) {
  std::vector<int64_t> x_dims_vec = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims_vec = phi::vectorize(y.dims());
  std::vector<int64_t>::const_iterator f1 = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l1 = x_dims_vec.end() - 2;
  std::vector<int64_t> x_dims_vec_cut(f1, l1);

  std::vector<int64_t>::const_iterator f2 = y_dims_vec.begin();
  std::vector<int64_t>::const_iterator l2 = y_dims_vec.end() - 2;
  std::vector<int64_t> y_dims_vec_cut(f2, l2);

  std::vector<int64_t> expand_batch_portion =
      get_broadcast_batch_portion(x_dims_vec_cut, y_dims_vec_cut);
  std::vector<int64_t> x_expand_size({expand_batch_portion});
  x_expand_size.insert(x_expand_size.end(),
                       {x_dims_vec[static_cast<int>(x_dims_vec.size()) - 2],
                        x_dims_vec[static_cast<int>(x_dims_vec.size()) - 1]});
  std::vector<int64_t> y_expand_size({expand_batch_portion});
  y_expand_size.insert(y_expand_size.end(),
                       {y_dims_vec[static_cast<int>(y_dims_vec.size()) - 2],
                        y_dims_vec[static_cast<int>(y_dims_vec.size()) - 1]});

  return std::make_tuple(x_expand_size, y_expand_size);
}

template <typename Context, typename T>
static void linalg_solve(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  phi::funcs::MatrixSolveFunctor<Context, T> mat_solve;

  // input y can be vector or matrix
  // but need to be unsqueezed if y is a vector
  bool is_vector = false;
  is_vector = is_vector_rhs(x, y);

  Tensor tmp_y;
  if (is_vector) {
    dev_ctx.Alloc(&tmp_y, y.dtype());

    phi::Unsqueeze<T, Context>(dev_ctx, y, {-1}, &tmp_y, nullptr);
  } else {
    tmp_y.Resize(y.dims());
    dev_ctx.Alloc(&tmp_y, y.dtype());

    phi::Copy(dev_ctx, y, dev_ctx.GetPlace(), false, &tmp_y);
  }

  Tensor tmp_x;
  tmp_x.Resize(x.dims());
  dev_ctx.Alloc(&tmp_x, x.dtype());
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &tmp_x);

  std::vector<int64_t> x_broadcast_dims;
  std::vector<int64_t> y_broadcast_dims;
  std::tie(x_broadcast_dims, y_broadcast_dims) =
      get_broadcast_dims(tmp_x, tmp_y);

  Tensor tmp_x_bc;

  phi::ExpandAsKernel<T, Context>(
      dev_ctx, tmp_x, nullptr, convert_to_int_vec(x_broadcast_dims), &tmp_x_bc);

  Tensor tmp_y_bc;
  phi::ExpandAsKernel<T, Context>(
      dev_ctx, tmp_y, nullptr, convert_to_int_vec(y_broadcast_dims), &tmp_y_bc);

  auto x_dim = x.dims();
  auto y_dim = y.dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();

  if (is_vector) {                 // vector case
    out->Resize(tmp_y_bc.dims());  // out.unsqueeze(-1)
    mat_solve(dev_ctx, tmp_x_bc, tmp_y_bc, out);

    Tensor out_tmp;
    out_tmp.Resize(out->dims());
    out_tmp = *out;

    phi::Squeeze<T, Context>(dev_ctx, out_tmp, {-1}, out);
  } else {
    PADDLE_ENFORCE_EQ(
        x_dim[x_dim_size - 1],
        y_dim[y_dim_size - 2],
        phi::errors::InvalidArgument(
            "Matrix X1 with dimension greater than 2 and any matrix Y1,"
            "the matrix X1's width must be equal with matrix Y1's "
            "height. But received X's shape = [%s], X1's shape = [%s], X1's "
            "width = %s; Y's shape = [%s], Y1's shape = [%s], Y1's height = "
            "%s.",
            x_dim,
            x_dim,
            x_dim[x_dim_size - 1],
            y_dim,
            y_dim,
            y_dim[y_dim_size - 2]));
    mat_solve(dev_ctx, tmp_x_bc, tmp_y_bc, out);
  }
}

template <typename T, typename Context>
void SolveKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  linalg_solve<Context, T>(dev_ctx, x, y, out);
}

}  // namespace phi
