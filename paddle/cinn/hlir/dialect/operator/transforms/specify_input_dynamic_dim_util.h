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

#pragma once

#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace cinn {
namespace dialect {
namespace ir {
void SpecifyInputDynamicDim(
    pir::Program* program,
    const std::vector<pir::InputDynamicDimSpec>& input_dynamic_dim_spec);
void SpecifyInputDynamicDimFromFile(pir::Program* program,
                                    std::string filepath);
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
