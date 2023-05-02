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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {
namespace funcs {

inline void SetXShape(const DenseTensor &x, DenseTensor *xshape) {
  const auto &in_dims = x.meta().dims;
  std::vector<int64_t> xshape_dims(in_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    xshape_dims[i + 1] = in_dims[i];
  }
  xshape->ResizeAndAllocate(phi::make_ddim(xshape_dims));
  xshape->ResetLoD(x.meta().lod);
}

inline void GetBroadcastDimsArrays(const DDim &x_dims,
                                   const DDim &y_dims,
                                   int *x_dims_array,
                                   int *y_dims_array,
                                   int *out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      phi::errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));

  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true,
        phi::errors::InvalidArgument(
            "Broadcast dimension mismatch. Operands could "
            "not be broadcast together with the shape of X = [%s] and "
            "the shape of Y = [%s]. Received [%d] in X is not equal to "
            "[%d] in Y at i:%d.",
            x_dims,
            y_dims,
            x_dims_array[i],
            y_dims_array[i],
            i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

inline void GetPrePostNumel(
    const DDim &dim, int axis, int *pre, int *n, int *post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

static DDim ExtendDims2Rank(const DDim &in_dims, int rank) {
  if (in_dims.size() == rank) {
    return in_dims;
  }
  std::vector<int64_t> shapes(rank, 1);
  for (int i = in_dims.size() - 1, j = rank - 1; i >= 0; --i, --j) {
    shapes[j] = in_dims[i];
  }
  return make_ddim(shapes);
}

template <size_t D>
static void GetBroadcastDims(const DDim &in_dims,
                             const DDim &out_dims,
                             Eigen::DSizes<int, D> *bcast_dims) {
  for (size_t i = 0; i < D; ++i) {
    if (in_dims[i] == out_dims[i]) {
      (*bcast_dims)[i] = 1;
    } else {
      (*bcast_dims)[i] = std::max(in_dims[i], out_dims[i]);
    }
  }
}

inline bool CheckDims(const DDim &dims_x, const DDim &dims_y) {
  if (dims_x.size() != dims_y.size()) {
    return false;
  }
  for (int i = 0; i < dims_x.size(); i++) {
    if (dims_x[i] != dims_y[i]) {
      return false;
    }
  }
  return true;
}

// Just For Matrix OP, for example:
// x's dim = [5, 3, 2, M, M] ; y's dim = [3, 1, M, N]
// out [5, 3, 2], which is batch_size of matrix
static inline std::vector<int64_t> MatrixGetBroadcastBatchPortion(
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

// Just For Matrix OP, for example:
// x's dim = [5, 3, 2, M, M] ; y's dim = [3, 1, M, N]
// out shoule be [5, 3, 2, M, M] + [5, 3, 2, M, N], and [5, 3, 2] is
// batch_size of matrix
static inline std::tuple<std::vector<int64_t>, std::vector<int64_t>>
MatrixGetBroadcastDims(const DenseTensor &x, const DenseTensor &y) {
  std::vector<int64_t> x_dims_vec = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims_vec = phi::vectorize(y.dims());

  std::vector<int64_t>::const_iterator f1 = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l1 = x_dims_vec.end() - 2;
  std::vector<int64_t> x_dims_vec_cut(f1, l1);

  std::vector<int64_t>::const_iterator f2 = y_dims_vec.begin();
  std::vector<int64_t>::const_iterator l2 = y_dims_vec.end() - 2;
  std::vector<int64_t> y_dims_vec_cut(f2, l2);

  std::vector<int64_t> expand_batch_portion =
      MatrixGetBroadcastBatchPortion(x_dims_vec_cut, y_dims_vec_cut);

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

inline DDim GetOutputDims(const DDim &s_dims, const DDim &l_dims) {
  if (s_dims.size() > l_dims.size()) {
    return GetOutputDims(l_dims, s_dims);
  }
  std::vector<int64_t> shapes = phi::vectorize<int64_t>(l_dims);
  for (int i = s_dims.size() - 1, j = l_dims.size() - 1; i >= 0; --i, --j) {
    int64_t s = s_dims[i];
    int64_t l = l_dims[j];
    if (s != l) {
      if (l == 1) {
        shapes[j] = s;
      } else if (s != 1) {
        PADDLE_THROW(errors::InvalidArgument(
            "The shape of tensor a %s:%d must match shape of tensor b "
            "%s:%d.",
            s_dims.to_str(),
            i,
            l_dims.to_str(),
            j));
      }
    }
  }
  return phi::make_ddim(shapes);
}

inline int64_t CalStride(phi::DDim dim) {
  int rank = dim.size();
  int64_t dimsum = 1;
  int64_t strides = 0;
  for (int i = rank - 1; i >= 0; i--) {
    strides += dimsum;
    dimsum *= dim[i];
  }
  return strides;
}

}  // namespace funcs
}  // namespace phi
