// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Pre-Processing function.
// The function here will be called by the functions in
// paddle/fluid/pybind/static_op_function.cc and
// paddle/fluid/pybind/eager_op_function.cc. Mainly used to customize the
// processing of parameters originally done in the Python API
#include "paddle/fluid/pybind/arg_pre_process.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {
void FlattenPreProcess_Common(int x_dim, int* start_axis, int* stop_axis) {
  if (x_dim == 0) {
    if (!(*start_axis == 0 || *start_axis == -1)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The start_axis should be int, and should be 0 or -1 when the input "
          "tensor is a 0-D-Tensor"));
    }
    if (!(*stop_axis == 0 || *stop_axis == -1)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The stop_axis should be int, and should be 0 or -1 when the input "
          "tensor is a 0-D-Tensor"));
    }
  } else {
    if (*start_axis > x_dim - 1 || *start_axis < -x_dim) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The start_axis should be a int, and in range [-rank(x), rank(x))"));
    }
    if (*stop_axis > x_dim - 1 || *stop_axis < -x_dim) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The stop_axis should be a int, and in range [-rank(x), rank(x))"));
    }
    if (*start_axis < 0) {
      *start_axis = *start_axis + x_dim;
    }
    if (*stop_axis < 0) {
      *stop_axis = *stop_axis + x_dim;
    }
    if (*start_axis > *stop_axis) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The stop_axis should be larger than stat_axis"));
    }
  }
}

void FlattenPreProcess(Tensor* x, int* start_axis, int* stop_axis) {
  int x_dim = x->dims().size();
  FlattenPreProcess_Common(x_dim, start_axis, stop_axis);
}
void FlattenPreProcess(Value* x, int* start_axis, int* stop_axis) {
  std::vector<int64_t> x_shape = pir::GetShapeFromValue(*x);
  int x_dim = x_shape.size();
  FlattenPreProcess_Common(x_dim, start_axis, stop_axis);
}

void RollPreProcess(Tensor* x, IntArray* shifts, IntVector* axis) {
  int64_t len_origin_shape = x->dims().size();
  if (axis != NULL) {
    int64_t axis_len = axis->size();
    for (int64_t i = 0; i < axis_len; i++) {
      PADDLE_ENFORCE_EQ(
          ((*axis)[i] < len_origin_shape && (*axis)[i] >= -len_origin_shape),
          true,
          common::errors::InvalidArgument("axis is out of range, it should be "
                                          "in range [%d, %d), but received %ld",
                                          -len_origin_shape,
                                          len_origin_shape,
                                          (*axis)[i]));
    }
  } else {
    axis = new IntVector();
  }
}
void RollPreProcess(Value* x, Value* shifts, IntVector* axis) {
  std::vector<int64_t> x_shape = pir::GetShapeFromValue(*x);
  int64_t len_origin_shape = x_shape.size();
  if (axis != NULL) {
    int64_t axis_len = axis->size();
    for (int64_t i = 0; i < axis_len; i++) {
      PADDLE_ENFORCE_EQ(
          ((*axis)[i] < len_origin_shape && (*axis)[i] >= -len_origin_shape),
          true,
          common::errors::InvalidArgument("axis is out of range, it should be "
                                          "in range [%d, %d), but received %ld",
                                          -len_origin_shape,
                                          len_origin_shape,
                                          (*axis)[i]));
    }
  } else {
    axis = new IntVector();
  }
}
}  // namespace pybind

}  // namespace paddle
