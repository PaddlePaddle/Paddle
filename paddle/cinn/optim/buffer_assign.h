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

#include <map>
#include <set>
#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace optim {

/**
 * Assign buffer for tensors those are not marked as compute_inline.
 * @param expr
 * @param stages The stage map.
 */
std::map<std::string, ir::Tensor> InitialAssignBuffer(
    Expr* expr,
    poly::StageMap stages,
    const std::map<std::string, ir::Tensor>& all_tensor_map,
    const common::Graph* comp_graph,
    const std::set<std::string>& temp_tensor_names);

}  // namespace optim
}  // namespace cinn
