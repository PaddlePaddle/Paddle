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
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {

namespace pybind {
using Tensor = paddle::Tensor;
using Value = pir::Value;
using IntArray = paddle::experimental::IntArray;
using IntVector = std::vector<int64_t>;

void FlattenPreProcess(Tensor* x, int* start_axis, int* stop_axis);
void FlattenPreProcess(Value* x, int* start_axis, int* stop_axis);

void RollPreProcess(Tensor* x, IntArray* shifts, IntVector* axis);
void RollPreProcess(Value* x, Value* shifts, IntVector* axis);

}  // namespace pybind

}  // namespace paddle
