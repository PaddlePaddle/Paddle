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

pir::OpResult builtin_combine(const std::vector<pir::Value>& x);

std::vector<pir::OpResult> add_n_grad(const std::vector<pir::Value>& inputs,
                                      const pir::Value& out_grad);

pir::OpResult zeros_like(const pir::Value& x,
                         phi::DataType dtype = phi::DataType::UNDEFINED,
                         const Place& place = {});

pir::OpResult get_parameter(const std::string& name);

void set_parameter(const pir::Value& parameter, const std::string& name);

pir::OpResult embedding_grad(const pir::Value& x,
                             const pir::Value& weight,
                             const pir::Value& out_grad,
                             int64_t padding_idx = -1,
                             bool sparse = false);

pir::OpResult split_with_num_grad(const std::vector<pir::Value>& out_grad,
                                  int axis);

pir::OpResult split_with_num_grad(const std::vector<pir::Value>& out_grad,
                                  const pir::Value& axis);
}  // namespace dialect
}  // namespace paddle
