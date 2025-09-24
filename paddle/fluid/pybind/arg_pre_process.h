// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include <vector>
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/utils/optional.h"
namespace paddle {

namespace pybind {
using Tensor = paddle::Tensor;
using Value = pir::Value;
using IntArray = paddle::experimental::IntArray;
using IntVector = std::vector<int64_t>;

void ExpandAsPreProcess(paddle::Tensor* x,
                        paddle::optional<paddle::Tensor>* y,
                        std::vector<int64_t>* target_shape);
void ExpandAsPreProcess(Value* x,
                        paddle::optional<pir::Value>* y,
                        std::vector<int64_t>* target_shape);
void RollPreProcess(Tensor* x, IntArray* shifts, IntVector* axis);
void RollPreProcess(Value* x, Value* shifts, IntVector* axis);

void LogsumexpPreProcess(Tensor* x, std::vector<int>* axis, bool* reduce_all);
void LogsumexpPreProcess(Value* x, std::vector<int>* axis, bool* reduce_all);

void SumPreProcess(Tensor* x, IntArray* axis);
void SumPreProcess(Value* x, Value* axis);
}  // namespace pybind

}  // namespace paddle
