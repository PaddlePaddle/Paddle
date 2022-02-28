// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <vector>
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"

namespace paddle {
namespace distributed {
using Tensor = paddle::experimental::Tensor;

std::vector<std::vector<size_t>> Eager_AssignGroupBySize(
    const std::vector<Tensor>, const std::vector<bool>& is_sparse_gradient,
    const std::vector<size_t>& group_size_limits,
    const std::vector<int64_t>& tensor_indices = {});

}  //  namespace distributed
}  //  namespace paddle
