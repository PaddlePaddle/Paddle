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
#include "paddle/pten/infershape/unary.h"
#include "paddle/pten/kernels/functions/lod_utils.h"

namespace pten {

DenseTensorMeta UnchangedInferShape(const DenseTensorMeta& x_meta) {
  return x_meta;
}

DenseTensorMeta ReductionInferShape(const DenseTensorMeta& x_meta) {
  const auto& out_dims = paddle::framework::make_ddim({1});
  DenseTensorMeta return_meta(x_meta.type, out_dims, x_meta.layout);
  return return_meta;
}

DenseTensorMeta FlattenInferShape(const DenseTensorMeta& x_meta,
                                  int start_axis,
                                  int stop_axis) {
  auto& x_dims = x_meta.dims;
  int in_dims_size = x_dims.size();
  if (start_axis < 0) {
    start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    stop_axis = stop_axis + in_dims_size;
  }
  PADDLE_ENFORCE_GE(stop_axis,
                    start_axis,
                    paddle::platform::errors::InvalidArgument(
                        "The stop_axis should be greater"
                        "than or equal to start_axis."));

  int64_t outer = 1;
  std::vector<int32_t> out_shape;
  out_shape.reserve(in_dims_size - stop_axis + start_axis);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(x_dims[i]);
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    if (x_dims[i] == -1 || outer == -1) {
      outer = -1;
    } else {
      outer *= x_dims[i];
    }
  }
  out_shape.push_back(outer);
  for (int i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(x_dims[i]);
  }
  const auto& out_dims = paddle::framework::make_ddim(out_shape);
  DenseTensorMeta return_meta(x_meta.type, out_dims, x_meta.layout);

  if (x_dims[0] == return_meta.dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    return_meta.lod = x_meta.lod;
  }

  return return_meta;
}

DenseTensorMeta ConcatInferShap(const std::vector<DenseTensorMeta>& x_metas,
                                int axis) {
  std::vector<pten::DDim> inputs_dims;
  inputs_dims.reserve(x_metas.size());
  for (size_t i = 0; i < x_metas.size(); ++i) {
    inputs_dims.emplace_back(x_metas[i].dims);
  }

  const size_t inputs_num = inputs_dims.size();
  PADDLE_ENFORCE_GT(
      inputs_num,
      static_cast<size_t>(0),
      paddle::platform::errors::InvalidArgument(
          "The number of input tensors in concat op should > 0. But "
          "received inputs' length is 0."));
  if (inputs_num == 1) {
    VLOG(3) << "Warning: concat op have only one input, may waste memory";
  }

  size_t axis1 = pten::ComputeAxis(axis, inputs_dims[0].size());

  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; ++i) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(),
                      out_dims.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The shape of input[0] and input[%d] "
                          "is expected to be equal."
                          "But received input[0]'s shape = "
                          "[%s], input[%d]'s shape = [%s].",
                          i,
                          inputs_dims[0],
                          i,
                          inputs_dims[i]));
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis1) {
        out_dims[axis] += inputs_dims[i][j];
      } else {
        bool check_shape = (inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0);
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j],
                            inputs_dims[i][j],
                            paddle::platform::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j,
                                i,
                                inputs_dims[0],
                                i,
                                inputs_dims[i]));
        }
      }
    }
  }
  DenseTensorMeta out_meta(x_metas[0].type, out_dims, x_metas[0].layout);
  return out_meta;
}

}  // namespace pten
