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
ir::OpResult add_n(std::vector<ir::OpResult> x);

ir::OpResult mean(ir::OpResult x,
                  std::vector<int64_t> axis = {},
                  bool keepdim = false);

ir::OpResult sum(ir::OpResult x,
                 std::vector<int64_t> axis = {},
                 phi::DataType dtype = phi::DataType::UNDEFINED,
                 bool keepdim = false);

ir::OpResult divide(ir::OpResult x, ir::OpResult y);

ir::OpResult full(std::vector<int64_t> shape,
                  float value,
                  phi::DataType dtype = phi::DataType::FLOAT32,
                  phi::Place place = phi::CPUPlace());

ir::OpResult tanh_grad(ir::OpResult out, ir::OpResult grad_out);

ir::OpResult mean_grad(ir::OpResult x,
                       ir::OpResult out_grad,
                       const std::vector<int64_t>& axis = {},
                       bool keepdim = false,
                       bool reduce_all = false);
}  // namespace dialect
}  // namespace paddle
