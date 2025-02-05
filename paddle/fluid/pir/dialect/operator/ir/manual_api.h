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

#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/platform/tensorrt/engine_params.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace dialect {

pir::Value builtin_combine(const std::vector<pir::Value>& x);
std::vector<pir::Value> builtin_split(const pir::Value& x);

std::vector<pir::Value> add_n_grad(const std::vector<pir::Value>& inputs,
                                   const pir::Value& out_grad);

pir::Value zeros_like(const pir::Value& x,
                      phi::DataType dtype = phi::DataType::UNDEFINED,
                      const Place& place = {});

pir::Value parameter(const std::string& name);

void set_parameter(const pir::Value& parameter, const std::string& name);

void update_parameter(const pir::Value& parameter, const std::string& name);

void shadow_output(const pir::Value& persist_value, const std::string& name);

pir::Value embedding_grad(const pir::Value& x,
                          const pir::Value& weight,
                          const pir::Value& out_grad,
                          int64_t padding_idx = -1,
                          bool sparse = false);

pir::Value split_with_num_grad(const std::vector<pir::Value>& out_grad,
                               int axis);

pir::Value split_with_num_grad(const std::vector<pir::Value>& out_grad,
                               const pir::Value& axis);

pir::Value ones(const std::vector<int64_t>& shape,
                phi::DataType dtype = phi::DataType::FLOAT32,
                const Place& place = phi::CPUPlace());

pir::Value full(const std::vector<int64_t>& shape,
                double real,
                double imag,
                phi::DataType dtype = phi::DataType::FLOAT32,
                const Place& place = phi::CPUPlace());

pir::Value ones_like(pir::Value x_,
                     phi::DataType dtype = phi::DataType::UNDEFINED,
                     const Place& place = {});

pir::Value zeros(const std::vector<int64_t>& shape,
                 phi::DataType dtype = phi::DataType::FLOAT32,
                 const Place& place = phi::CPUPlace());

pir::Value create_array(phi::DataType dtype);

pir::Value create_array_like(pir::Value input, float value);

pir::Value array_length(pir::Value x);

pir::Value array_read(pir::Value array, pir::Value i);

pir::Value fetch(pir::Value value, std::string name, int col);

pir::Value array_write_(pir::Value array, pir::Value x, pir::Value i);

std::tuple<pir::Value, pir::Value> array_to_tensor(pir::Value x,
                                                   int axis,
                                                   bool use_stack);

pir::Value tensor_to_array(pir::Value x,
                           pir::Value out_grad,
                           int axis,
                           bool use_stack);

pir::Value add_n_array(const std::vector<pir::Value>& inputs);

pir::Value slice_array(pir::Value input, pir::Value starts, pir::Value ends);

pir::Value slice_array_dense(pir::Value input, pir::Value starts);

pir::Value assign(const pir::Value& x);

std::tuple<pir::Value, pir::Value> fused_gemm_epilogue(pir::Value x,
                                                       pir::Value y,
                                                       pir::Value bias,
                                                       bool trans_x,
                                                       bool trans_y,
                                                       std::string activation);

std::tuple<pir::Value, pir::Value, pir::Value> fused_gemm_epilogue_grad(
    pir::Value x,
    pir::Value y,
    paddle::optional<pir::Value> reserve_space,
    pir::Value out_grad,
    bool trans_x,
    bool trans_y,
    std::string activation_grad);

pir::Value array_pop(pir::Value input, int index);

std::vector<pir::Value> tensorrt_engine(
    const std::vector<pir::Value>& inputs,
    paddle::platform::EngineParams trt_params,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::vector<std::vector<int64_t>> outputs_shape,
    std::vector<phi::DataType> outputs_dtype,
    const std::string& converter_debug_info);

pir::Operation* share_var(const std::vector<pir::Value>& x);
}  // namespace dialect
}  // namespace paddle
