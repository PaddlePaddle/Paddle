/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {

inline Tensor ReshapeToMatrix(const Tensor& src, int num_col_dims) {
  int rank = src.dims().size();
  PADDLE_ENFORCE_GE(
      rank, 2, platform::errors::InvalidArgument(
                   "'ReshapeToMatrix()' is only used for flatten high rank "
                   "tensors to matrixs. The dimensions of Tensor must be "
                   "greater or equal than 2. "
                   "But received dimensions of Tensor is %d",
                   rank));
  if (rank == 2) {
    return src;
  }
  Tensor res;
  res.ShareDataWith(src);
  res.Resize(flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

}  // namespace framework
}  // namespace paddle
