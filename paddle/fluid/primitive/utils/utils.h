// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace primitive {

template <typename T>
void set_output(const Tensor& x_tmp, Tensor* x);

// This fucction compute unsqueeze dims for reshape to replace unsqueeze.
static std::vector<int64_t> get_unsqueeze_dims(
    const Tensor& origin, const std::vector<int64_t>& axis) {
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size() + axis.size();
  std::vector<int64_t> result;
  size_t j = 0, k = 0;
  for (size_t i = 0; i < total_shape_size; ++i) {
    if (j < axis.size() && axis[j] == int64_t(i)) {
      result.push_back(1);
      j++;
    } else {
      PADDLE_ENFORCE_LT(
          k,
          origin_dims.size(),
          platform::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                       "elements in origin_dims[%lu].",
                                       k,
                                       origin_dims.size()));
      result.push_back(origin_dims[k]);
      k++;
    }
  }
  return result;
}

// These method don't need to be specified
static phi::DDim get_reduce_dims_from_out(const phi::DDim& dout_dims,
                                          const phi::DDim& in_dims) {
  std::vector<int64_t> result;
  int bat = dout_dims.size() - in_dims.size();
  for (int i = 0; i < bat; ++i) {
    result.push_back(i);
  }
  for (int i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] == 1) {
      result.push_back(i + bat);
    } else {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          dout_dims[i + bat],
          platform::errors::InvalidArgument(
              "ReduceDims dimension mismatch. Operands could "
              "not be broadcast together with the shape of dout = [%s] and "
              "the shape of in_dims = [%s]. Received [%d] in X is not equal to "
              "[%d] in Y at i:%d.",
              dout_dims,
              in_dims,
              dout_dims[i + bat],
              in_dims[i],
              i));
    }
  }
  return phi::make_ddim(result);
}

static phi::DDim get_reduce_dims(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims) {
  auto out_dims = paddle::operators::details::BroadcastTwoDims(x_dims, y_dims);
  return get_reduce_dims_from_out(out_dims, x_dims);
}

void SetEmptyGrad(const std::vector<std::vector<Tensor>>& outputs,
                  const std::vector<std::vector<bool>>& stop_gradients);

std::vector<std::vector<Tensor>> ConstructVjpResultByStopGradients(
    const std::vector<std::vector<Tensor>>& outputs,
    const std::vector<std::vector<bool>>& stop_gradients);

}  // namespace primitive
}  // namespace paddle
