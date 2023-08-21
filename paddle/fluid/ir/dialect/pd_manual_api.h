// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/ir/core/value.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace dialect {
std::vector<ir::OpResult> split(ir::OpResult x,
                                const std::vector<int64_t>& sections,
                                int axis);

std::vector<ir::OpResult> concat_grad(std::vector<ir::OpResult> x,
                                      ir::OpResult out_grad,
                                      ir::OpResult axis);

ir::OpResult split_grad(std::vector<ir::OpResult> out_grads, ir::OpResult axis);

}  // namespace dialect
}  // namespace paddle
