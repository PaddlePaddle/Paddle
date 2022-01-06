// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "paddle/extension.h"

#define CHECK_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> SimpleSliceFunction(const paddle::Tensor& x,
                                                int64_t begin_index,
                                                int64_t end_index) {
  return {x.slice(begin_index, end_index)};
}

std::vector<std::vector<int64_t>> SimpleSliceInferShape(
    const std::vector<int64_t>& x_shape,
    int64_t begin_index,
    int64_t end_index) {
  PD_CHECK(begin_index > 0, "The begin index is out of bound.");
  PD_CHECK(end_index > 0, "The end index must is out of bound.");
  PD_CHECK(begin_index < end_index,
           "The begin index is greater than end index.");
  auto out_shape = x_shape;
  out_shape[0] = end_index - begin_index;
  return {out_shape};
}

PD_BUILD_OP(custom_simple_slice)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"begin_index: int64_t", "end_index: int64_t"})
    .SetKernelFn(PD_KERNEL(SimpleSliceFunction))
    .SetInferShapeFn(PD_INFER_SHAPE(SimpleSliceInferShape));
