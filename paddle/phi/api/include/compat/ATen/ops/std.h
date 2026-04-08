// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/OptionalArrayRef.h>
#include <optional>
#include <vector>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"

namespace at {

// Internal implementation for std (standard deviation = sqrt(variance))
inline Tensor std_impl(const Tensor& self,
                       const std::vector<int64_t>& dims_vec,
                       double correction_value,
                       bool keepdim) {
  // Validate dimensions before processing
  int64_t ndim = self.dim();
  for (int64_t d : dims_vec) {
    int64_t dim_idx = d < 0 ? d + ndim : d;
    if (dim_idx < 0 || dim_idx >= ndim) {
      PD_CHECK(false,
               "Dimension out of range (expected to be in range of [",
               -ndim,
               ", ",
               ndim - 1,
               "], but got ",
               d,
               ")");
    }
  }
  phi::IntArray dims_int_array(dims_vec);
  paddle::Tensor tensor = self._PD_GetInner();

  paddle::Tensor mean_tensor;
  if (dims_vec.empty()) {
    mean_tensor = paddle::experimental::mean(
        tensor, phi::IntArray(std::vector<int64_t>{}), true);
  } else {
    mean_tensor = paddle::experimental::mean(tensor, dims_int_array, true);
  }

  paddle::Tensor diff = paddle::experimental::subtract(tensor, mean_tensor);
  paddle::Tensor diff_squared = paddle::experimental::multiply(diff, diff);

  paddle::Tensor sum_squared_diff;
  if (dims_vec.empty()) {
    sum_squared_diff =
        paddle::experimental::sum(diff_squared,
                                  phi::IntArray(std::vector<int64_t>{}),
                                  diff_squared.dtype(),
                                  keepdim);
  } else {
    sum_squared_diff = paddle::experimental::sum(
        diff_squared, dims_int_array, diff_squared.dtype(), keepdim);
  }

  int64_t n = tensor.numel();
  if (!dims_vec.empty()) {
    n = 1;
    for (int64_t d : dims_vec) {
      int64_t dim_idx = d < 0 ? d + tensor.dims().size() : d;
      if (dim_idx >= 0 &&
          dim_idx < static_cast<int64_t>(tensor.dims().size())) {
        n *= tensor.dims()[dim_idx];
      }
    }
  }

  double corrected_n = static_cast<double>(n) - correction_value;
  if (corrected_n <= 0.0) {
    corrected_n = static_cast<double>(n);
  }

  std::vector<int64_t> result_shape_vec;
  for (int64_t i = 0; i < sum_squared_diff.dims().size(); ++i) {
    result_shape_vec.push_back(sum_squared_diff.dims()[i]);
  }
  paddle::Tensor correction_scalar =
      paddle::experimental::full(phi::IntArray(result_shape_vec),
                                 phi::Scalar(corrected_n),
                                 sum_squared_diff.dtype(),
                                 sum_squared_diff.place());
  paddle::Tensor variance =
      paddle::experimental::divide(sum_squared_diff, correction_scalar);

  paddle::Tensor result = paddle::experimental::sqrt(variance);

  return Tensor(result);
}

}  // namespace at

namespace at {

inline Tensor Tensor::std(int dim) const {
  // Call the OptionalIntArrayRef version
  return std(at::OptionalIntArrayRef(dim), true, false);
}

inline Tensor Tensor::std(bool unbiased) const {
  std::vector<int64_t> empty_dims;
  double correction = unbiased ? 1.0 : 0.0;
  return std_impl(*this, empty_dims, correction, false);
}

inline Tensor Tensor::std(at::OptionalIntArrayRef dim,
                          bool unbiased,
                          bool keepdim) const {
  double correction = unbiased ? 1.0 : 0.0;
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return std_impl(*this, dims_vec, correction, keepdim);
}

inline Tensor Tensor::std(at::OptionalIntArrayRef dim,
                          const ::std::optional<at::Scalar>& correction,
                          bool keepdim) const {
  double correction_value = 1.0;
  if (correction.has_value()) {
    const at::Scalar& scalar = correction.value();
    correction_value = scalar.to<double>();
  }
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  return std_impl(*this, dims_vec, correction_value, keepdim);
}

}  // namespace at
