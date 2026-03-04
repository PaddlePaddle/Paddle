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
using Value = pir::Value;
using IntArray = paddle::experimental::IntArray;
using Scalar = paddle::experimental::Scalar;
using IntVector = std::vector<int64_t>;

void ExpandAsPreProcess(Tensor* x,
                        paddle::optional<Tensor>* y,
                        std::vector<int64_t>* target_shape);
void ExpandAsPreProcess(Value* x,
                        paddle::optional<pir::Value>* y,
                        std::vector<int64_t>* target_shape);
void RollPreProcess(Tensor* x, IntArray* shifts, IntVector* axis);
void RollPreProcess(Value* x, Value* shifts, IntVector* axis);

void BinCountPreProcess(Tensor* x,
                        paddle::optional<Tensor>* weights,
                        Scalar* minlength);
void BinCountPreProcess(Value* x,
                        paddle::optional<Value>* weights,
                        Value* minlength);

void LogsumexpPreProcess(Tensor* x, std::vector<int>* axis, bool* reduce_all);
void LogsumexpPreProcess(Value* x, std::vector<int>* axis, bool* reduce_all);

void SumPreProcess(Value* x, Value* axis);
void IsClosePreProcess(Value* x, Value* y, Value* rtol, Value* atol);
void AllClosePreProcess(Value* x, Value* y, Value* rtol, Value* atol);

void GridSamplePreProcess(Tensor* x,
                          Tensor* grid,
                          std::string* mode,
                          std::string* padding_mode,
                          bool* align_corners);
void GridSamplePreProcess(Value* x,
                          Value* grid,
                          std::string* mode,
                          std::string* padding_mode,
                          bool* align_corners);

// Addmm broadcast validation for dygraph
void AddmmPreProcess(Tensor* input, Tensor* x, Tensor* y);

// Addmm broadcast validation for static graph
void AddmmPreProcess(pir::Value* input, pir::Value* x, pir::Value* y);

// Baddbmm broadcast validation for dygraph
void BaddbmmPreProcess(Tensor* input, Tensor* x, Tensor* y);

// Baddbmm broadcast validation for static graph
void BaddbmmPreProcess(pir::Value* input, pir::Value* x, pir::Value* y);

// Renorm preprocessing: handle negative axis
void NegativeAxisPreProcess(Tensor* x, int* axis);
void NegativeAxisPreProcess(Value* x, int* axis);

void PixelShufflePreProcess(std::string* data_format);

// Inplace API broadcast validation for dygraph
void InplaceShapePreProcess(Tensor* x, Tensor* y);

// Inplace API broadcast validation for static graph
void InplaceShapePreProcess(pir::Value* x, pir::Value* y);

}  // namespace pybind
}  // namespace paddle
