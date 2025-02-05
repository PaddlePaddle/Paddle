// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

static paddle::Tensor Transpose(const paddle::Tensor& t,
                                int64_t dim0,
                                int64_t dim1) {
  int len = t.shape().size();
  dim0 = dim0 >= 0 ? dim0 : len + dim0;
  dim1 = dim1 >= 0 ? dim1 : len + dim1;
  PD_CHECK(dim0 >= 0 && dim0 < len,
           "dim0 not in range"
           "dim0:%d ,range:%d",
           dim0,
           len);
  PD_CHECK(dim1 >= 0 && dim1 < len,
           "dim1 not in range"
           "dim1:%d ,range:%d",
           dim1,
           len);
  std::vector<int> transpose_perm(len);
  std::iota(transpose_perm.begin(), transpose_perm.end(), 0);
  transpose_perm[dim0] = dim1;
  transpose_perm[dim1] = dim0;
  // maybe there is another way to avoid experiment api
  return paddle::experimental::transpose(t, transpose_perm);
}

std::vector<paddle::Tensor> ContiguousForward(paddle::Tensor& x) {  // NOLINT
  PD_CHECK(x.shape().size() == 2, "x must be a 2-d tensor.");

  x = x.contiguous();
  PD_CHECK(x.is_contiguous(), "Check failed !");

  auto non_contiguous_x = Transpose(x, 0, 1);
  PD_CHECK(!non_contiguous_x.is_contiguous(), "Check failed !");

  auto contiguous_x = non_contiguous_x.contiguous();
  PD_CHECK(contiguous_x.is_contiguous(), "Check failed !");

  return {contiguous_x};
}

PD_BUILD_OP(custom_contiguous)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ContiguousForward));
