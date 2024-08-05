// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <unordered_map>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {

ir::Tensor const_matrix(const std::vector<std::vector<float>>& input,
                        const std::string& name);

std::vector<std::vector<std::vector<float>>> get_winograd_val(
    const int& tile_size, const int& kernel_size);

std::vector<ir::Tensor> winograd_transform_matrices(const int& tile_size,
                                                    const int& kernel_size);

std::vector<int> GetFirstStepReduceShape(const std::vector<int>& shape,
                                         const std::vector<int>& axes,
                                         bool& inbound,  // NOLINT
                                         int& tail);     // NOLINT

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
