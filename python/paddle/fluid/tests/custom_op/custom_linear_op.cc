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

#include <iostream>
#include <vector>
#include "paddle/extension.h"

// The linear implemented here must be passed in bias
std::vector<paddle::Tensor> PhiLinearForward(const paddle::Tensor& x,
                                             const paddle::Tensor& weight,
                                             const paddle::Tensor& bias) {
  return {
      paddle::experimental::add(paddle::experimental::matmul(x, weight), bias)};
}

std::vector<std::vector<int64_t>> LinearInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  auto dims_x = x_shape;
  auto dims_y = weight_shape;
  auto ndims_x = x_shape.size();
  auto ndims_y = weight_shape.size();
  PD_CHECK(ndims_x > 0,
           "The Input(x) dims size must be greater than 0,"
           " but reviced dims size is 0. ");
  PD_CHECK(ndims_y > 0,
           "The Input(y) dims size must be greater than 0,"
           " but reviced dims size is 0. ");

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
  M = dims_x[ndims_x - 2];
  N = dims_y[ndims_y - 1];

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

  return {new_dims};
}

std::vector<paddle::DataType> LinearInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& weight_dtype,
    const paddle::DataType& bias_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(pten_linear)
    .Inputs({"X", "Weight", "Bias"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(PhiLinearForward))
    .SetInferShapeFn(PD_INFER_SHAPE(LinearInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LinearInferDtype));
