// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace op {

std::vector<ir::Tensor> ArgSort(const ir::Tensor& A,
                                const common::Target& target,
                                poly::StageMap stages,
                                const int& axis,
                                const bool& is_ascend,
                                const std::string& name);

std::vector<ir::Tensor> Sort(const ir::Tensor& A,
                             const common::Target& target,
                             poly::StageMap stages,
                             const int& axis,
                             const bool& is_ascend,
                             const std::string& name);

}  // namespace op
}  // namespace hlir
}  // namespace cinn
