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

std::vector<paddle::Tensor> ContiguousForward(paddle::Tensor& x) {  // NOLINT
  if (!x.is_contiguous()) {
    x = x.contiguous();
  }

  return {x};
}

PD_BUILD_OP(custom_contiguous)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ContiguousForward));
