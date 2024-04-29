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

#include <set>

#include "paddle/cinn/cinn.h"

namespace cinn {
namespace optim {

/**
 * Recursive expand the inlined tensors.
 * @param expr the expression to modify.
 * @param tensor_name name of the tensor to expand inline.
 * @param memo a memo to avoid duplicate expand.
 */
void ComputeInlineExpand(Expr* expr,
                         poly::StageMap stages,
                         std::map<std::string, ir::Tensor>* all_tensor_map);

}  // namespace optim
}  // namespace cinn
