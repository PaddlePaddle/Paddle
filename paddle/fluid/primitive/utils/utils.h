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

#include "paddle/common/ddim.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace primitive {

template <typename T>
void set_output(const Tensor& x_tmp, Tensor* x);

template <typename T>
void by_pass(const Tensor& x_tmp, Tensor* x);

// This function determine whether dtype is in [float16, bfloat16, uint16]
static bool is_half_dtype(const DataType& dtype) {
  if (dtype == phi::DataType::FLOAT16 || dtype == phi::DataType::BFLOAT16 ||
      dtype == phi::DataType::UINT16) {
    return true;
  } else {
    return false;
  }
}

// This function expands the dimension of origin Tensor based on the value of
// axis
static std::vector<int64_t> get_expand_dims(const Tensor& origin,
                                            const std::vector<int64_t>& axis) {
  std::vector<int64_t> result(origin.shape());
  for (size_t i = 0; i < axis.size(); ++i) {
    int64_t offset = axis[i];
    if (offset < 0) {
      offset += result.size() + 1;
    }

    PADDLE_ENFORCE_LE(
        offset,
        result.size(),
        platform::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                     "elements in origin_dims[%lu].",
                                     offset,
                                     result.size()));
    result.insert(result.begin() + offset, 1);
  }
  return result;
}

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

// This fucction compute unsqueeze dims for reshape to replace unsqueeze.
static std::vector<int64_t> get_squeeze_dims(const Tensor& origin,
                                             const std::vector<int64_t>& axis) {
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size();
  std::vector<int64_t> result;
  for (size_t i = 0; i < total_shape_size; ++i) {
    if (origin_dims[i] != 1) {
      result.push_back(origin_dims[i]);
    } else if (origin_dims[i] == 1 &&
               std::find(axis.begin(), axis.end(), int64_t(i)) == axis.end()) {
      result.push_back(1);
    } else {
      continue;
    }
  }
  return result;
}

static std::vector<int64_t> process_dims(const Tensor& origin,
                                         const std::vector<int64_t>& axis) {
  auto origin_dims = origin.shape();
  auto total_shape_size = origin_dims.size();
  std::vector<int64_t> result;
  auto axis_size = axis.size();
  if (axis_size == 0) {
    for (size_t i = 0; i < total_shape_size; ++i) {
      result.push_back(i);
    }
  } else {
    for (size_t i = 0; i < axis_size; ++i) {
      if (axis[i] < 0) {
        result.push_back(axis[i] + total_shape_size);
      } else {
        result.push_back(axis[i]);
      }
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
  return common::make_ddim(result);
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

static bool find_value(const std::vector<int64_t>& vec, int64_t value) {
  if (std::find(vec.begin(), vec.end(), value) != vec.end()) {
    return true;
  } else {
    return false;
  }
}

}  // namespace primitive
}  // namespace paddle
