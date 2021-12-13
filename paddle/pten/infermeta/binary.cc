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

// See Note [ Why still include the fluid headers? ]
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/kernels/hybird/general/elementwise_base.h"

namespace pten {

DenseTensorMeta DotInferMeta(const DenseTensorMeta& x_meta,
                             const DenseTensorMeta& y_meta) {
  auto x_dims = x_meta.dims;
  auto x_rank = static_cast<size_t>(x_dims.size());
  PADDLE_ENFORCE_EQ(true,
                    1 == x_rank || 2 == x_rank,
                    paddle::platform::errors::PreconditionNotMet(
                        "ShapeError: The dimensions of input tensor X (%s) "
                        "should be 1 or 2",
                        x_dims.to_str()));

  auto y_dims = y_meta.dims;
  PADDLE_ENFORCE_EQ(
      true,
      x_rank == (size_t)y_dims.size(),
      paddle::platform::errors::PreconditionNotMet(
          "ShapeError: The shape of input tensor Y: %s should match with "
          "input tenosr X: %s",
          y_dims.to_str(),
          x_dims.to_str()));
  bool shape_match = true;
  for (size_t i = 0; i < x_rank; ++i) {
    if (x_dims[i] != y_dims[i]) {
      shape_match = false;
      break;
    }
  }

  PADDLE_ENFORCE_EQ(true,
                    shape_match,
                    paddle::platform::errors::PreconditionNotMet(
                        "ShapeError: The shape of input tensor X: %s should "
                        "be exactly the same "
                        "with input tensor Y: %s",
                        x_dims.to_str(),
                        y_dims.to_str()));

  x_dims[x_dims.size() - 1] = 1;
  DenseTensorMeta return_meta(x_meta.dtype, x_dims, x_meta.layout);
  return return_meta;
}

DenseTensorMeta MatmulInferMeta(const DenseTensorMeta& x_meta,
                                const DenseTensorMeta& y_meta,
                                bool trans_x,
                                bool trans_y) {
  std::vector<int64_t> dims_x = paddle::framework::vectorize(x_meta.dims);
  std::vector<int64_t> dims_y = paddle::framework::vectorize(y_meta.dims);
  auto ndims_x = dims_x.size();
  auto ndims_y = dims_y.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M, N;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }
  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(dims_x.begin(), dims_x.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(dims_y.begin(), dims_y.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(dims_x[i], dims_y[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }
  if (x_broadcasted && y_broadcasted) {
    new_dims.push_back(1);
  }

  auto ddim_out = paddle::framework::make_ddim(new_dims);

  return {x_meta.dtype, ddim_out, x_meta.layout};
}

DenseTensorMeta ElementwiseInferMeta(const DenseTensorMeta& x_meta,
                                     const DenseTensorMeta& y_meta,
                                     int axis) {
  DenseTensorMeta return_meta(x_meta.dtype, x_meta.dims, x_meta.layout);
  if (x_meta.dims != y_meta.dims) {
    auto x_dims = x_meta.dims;
    auto y_dims = y_meta.dims;
    int max_dim = std::max(x_dims.size(), y_dims.size());
    if (x_dims.size() == y_dims.size()) {
      PADDLE_ENFORCE_EQ((axis == -1) || (axis == 0),
                        true,
                        paddle::platform::errors::InvalidArgument(
                            "axis should be -1 or 0 while the dimension of "
                            "tensor X (%s) is equal to the dimension of "
                            "tensor Y (%s), but received axis: %s",
                            x_dims.size(),
                            y_dims.size(),
                            axis));
    }
    PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The axis range must be [%s, %s), but axis is %s. "
                          "Please set the axis again.",
                          -1 * max_dim,
                          max_dim,
                          axis));
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    general::GetBroadcastDimsArrays(x_dims,
                                    y_dims,
                                    x_dims_array.data(),
                                    y_dims_array.data(),
                                    out_dims_array.data(),
                                    max_dim,
                                    axis);
    return_meta.dims = paddle::framework::make_ddim(out_dims_array);
  }
  return_meta.lod = x_meta.lod;
  return return_meta;
}

}  // namespace pten
