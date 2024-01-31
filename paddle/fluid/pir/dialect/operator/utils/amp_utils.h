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

#include "paddle/phi/common/data_type.h"
#include "paddle/pir/core/value.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace dialect {

phi::DataType GetPromoteType(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_tensors_vector,
    const phi::DataType& amp_dtype);

pir::Value Cast(const pir::Value& input, const phi::DataType& dst_dtype);

bool NeedCast(const pir::Value& value, const phi::DataType& dst_dtype);

pir::Value PirAmpAutoCast(const std::string& input_name,
                          const pir::Value& input,
                          const phi::DataType& dst_dtype,
                          const std::string& op_name);

paddle::optional<pir::Value> PirAmpAutoCast(
    const std::string& input_name,
    const paddle::optional<pir::Value>& input,
    const phi::DataType& dst_dtype,
    const std::string& op_name);

std::vector<pir::Value> PirAmpAutoCast(const std::string& inputs_name,
                                       const std::vector<pir::Value>& inputs,
                                       const phi::DataType& dst_dtype,
                                       const std::string& op_name);

paddle::optional<std::vector<pir::Value>> PirAmpAutoCast(
    const std::string& inputs_name,
    const paddle::optional<std::vector<pir::Value>>& inputs,
    const phi::DataType& dst_dtype,
    const std::string& op_name);

phi::DataType GetAmpDestDtype(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_values_vector);

}  // namespace dialect
}  // namespace paddle
