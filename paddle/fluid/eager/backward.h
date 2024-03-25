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
#include "paddle/phi/api/all.h"
#include "paddle/utils/test_macros.h"

namespace egr {

// Backward():
// tensors corresponds to those lived in the backward graph
// each grad_tensors[i] keeps the value for its corresponding tensors[i]
TEST_API void Backward(const std::vector<paddle::Tensor>& tensors,
                       const std::vector<paddle::Tensor>& grad_tensors,
                       bool retain_graph = false);

TEST_API std::vector<paddle::Tensor> Grad(
    const std::vector<paddle::Tensor>& tensors,
    const std::vector<paddle::Tensor>& inputs,
    const std::vector<paddle::Tensor>& grad_tensors = {},
    bool retain_graph = false,
    bool create_graph = false,
    bool only_inputs = false,
    bool allow_unused = false,
    const std::vector<paddle::Tensor>& no_grad_vars = {});

// Reserved for gradient()

}  // namespace egr
