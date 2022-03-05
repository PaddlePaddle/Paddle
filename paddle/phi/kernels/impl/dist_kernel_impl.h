// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <math.h>
#include <algorithm>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using ETensor = phi::EigenTensor<T, D, MajorType, IndexType>;

template <int Rank>
static void GetBraodcastDims(const phi::DDim& x_dims,
                             const phi::DDim& y_dims,
                             Eigen::DSizes<int, Rank>* x_bcast_dims,
                             Eigen::DSizes<int, Rank>* y_bcast_dims) {
  int bcast_dims_remainder = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      (*x_bcast_dims)[i] = 1;
      (*y_bcast_dims)[i] = x_dims[i] / y_dims[i];
      bcast_dims_remainder += x_dims[i] % y_dims[i];
    } else {
      (*y_bcast_dims)[i] = 1;
      (*x_bcast_dims)[i] = y_dims[i] / x_dims[i];
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
  }
  PADDLE_ENFORCE_EQ(bcast_dims_remainder,
                    0,
                    phi::errors::PreconditionNotMet(
                        "The input tensor of Op(dist) could not be broadcast, "
                        "X's shape is [%s], Y's shape is [%s].",
                        x_dims,
                        y_dims));
}

static phi::DDim GetNewDims(const phi::DDim& in_dims, int rank) {
  std::vector<int64_t> new_dims_vec(rank);
  if (in_dims.size() < rank) {
    for (int i = 0; i < rank - in_dims.size(); ++i) {
      new_dims_vec[i] = 1;
    }
    for (int i = 0; i < in_dims.size(); ++i) {
      new_dims_vec[i + rank - in_dims.size()] = in_dims[i];
    }
  } else {
    new_dims_vec = vectorize(in_dims);
  }
  return phi::make_ddim(new_dims_vec);
}

template <typename Context, typename T, int Rank>
static void DistFunction(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         float p,
                         DenseTensor* out) {
  if (out) {
    dev_ctx.template Alloc<T>(out);
  }
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  // new dims with same size as rank, e.g. (rank=3, (4, 3) => (1, 4, 3))
  phi::DDim x_new_dims = GetNewDims(x_dims, Rank);
  phi::DDim y_new_dims = GetNewDims(y_dims, Rank);

  auto x_t = ETensor<T, Rank>::From(x, x_new_dims);
  auto y_t = ETensor<T, Rank>::From(y, y_new_dims);
  auto out_t = ETensor<T, 1>::From(*out);
  auto& place = *dev_ctx.eigen_device();

  Eigen::DSizes<int, Rank> x_bcast_dims;
  Eigen::DSizes<int, Rank> y_bcast_dims;
  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  // p=0 means number of non-zero elements of (x-y)
  // p=inf means the maximum of |x-y|
  // p=-inf means the minimum of |x-y|
  // otherwise, Lp-norm = pow(sum(pow(|x-y|, p)), 1/p)
  if (p == 0) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) != y_t.broadcast(y_bcast_dims))
            .template cast<T>()
            .sum();
  } else if (p == INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .maximum();
  } else if (p == -INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .minimum();
  } else {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .pow(p)
            .sum()
            .pow(1.0 / p);
  }
}

template <typename T, typename Context>
void DistKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                float p,
                DenseTensor* out) {
  auto x_rank = x.dims().size();
  auto y_rank = y.dims().size();
  auto rank = std::max(x_rank, y_rank);
  PADDLE_ENFORCE_LE(rank,
                    6,
                    phi::errors::Unimplemented(
                        "Op(dist) only support tensors with no more than 6 "
                        "dimensions, but X's rank is %d, Y's rank is %d.",
                        x_rank,
                        y_rank));
  switch (rank) {
    case 1:
      DistFunction<Context, T, 1>(dev_ctx, x, y, p, out);
      break;
    case 2:
      DistFunction<Context, T, 2>(dev_ctx, x, y, p, out);
      break;
    case 3:
      DistFunction<Context, T, 3>(dev_ctx, x, y, p, out);
      break;
    case 4:
      DistFunction<Context, T, 4>(dev_ctx, x, y, p, out);
      break;
    case 5:
      DistFunction<Context, T, 5>(dev_ctx, x, y, p, out);
      break;
    case 6:
      DistFunction<Context, T, 6>(dev_ctx, x, y, p, out);
      break;
  }
}

}  // namespace phi
