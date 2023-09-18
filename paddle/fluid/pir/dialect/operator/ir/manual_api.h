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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/core/op_result.h"

namespace paddle {
namespace dialect {

pir::OpResult builtin_combine(std::vector<pir::OpResult> x);

std::vector<pir::OpResult> add_n_grad(std::vector<pir::OpResult> inputs,
                                      pir::OpResult out_grad);

pir::OpResult zeros_like(pir::OpResult x,
                         phi::DataType dtype = phi::DataType::UNDEFINED,
                         const Place& place = {});

pir::OpResult get_parameter(const std::string& name,
                            phi::DataType dtype,
                            const std::vector<int64_t>& shape);

void set_parameter(pir::OpResult parameter, const std::string& name);

pir::OpResult embedding_grad(pir::OpResult x,
                             pir::OpResult weight,
                             pir::OpResult out_grad,
                             int64_t padding_idx = -1,
                             bool sparse = false);

pir::OpResult split_with_num_grad(std::vector<pir::Value> out_grad, int axis);

pir::OpResult split_with_num_grad(std::vector<pir::Value> out_grad,
                                  pir::OpResult axis);
}  // namespace dialect
}  // namespace paddle
