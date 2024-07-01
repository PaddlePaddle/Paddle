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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace op {
std::vector<ir::Tensor> Argmax(const ir::Tensor &in_tensor,
                               const cinn::common::Target &target,
                               const int &axis,
                               const bool &keep_dims = false,
                               const std::string &name = "T_Argmax_out");
}  // namespace op
}  // namespace hlir
}  // namespace cinn
