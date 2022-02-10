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

#pragma once

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/pten/api/all.h"

namespace egr {

// run_backward():
// tensors corresponds to those lived in the backward graph
// each grad_tensors[i] keeps the value for its corresponding tensors[i]
void RunBackward(const std::vector<paddle::experimental::Tensor> &tensors,
                 const std::vector<paddle::experimental::Tensor> &grad_tensors,
                 bool retain_graph = false);

// Reserved for gradient()

}  // namespace egr
