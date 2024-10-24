// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * Lower lowerise the statements to LoweredFuncs.
 */

#pragma once
#include <string>
#include <vector>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/poly/schedule.h"

namespace cinn {
namespace lang {
using ir::Tensor;

ir::LoweredFunc LowerToAst(
    const std::string &name,
    const std::vector<Tensor> &tensor_args,
    ast_gen_ius::TensorGroup *tensor_group,
    const Target &target = cinn::common::DefaultHostTarget());

std::vector<ir::LoweredFunc> LowerToAstVec(
    const std::string &name,
    const std::vector<Tensor> &tensor_args,
    ast_gen_ius::TensorGroup *tensor_group,
    const Target &target = cinn::common::DefaultHostTarget());

std::vector<ir::Buffer> GetTempBuffers(
    const std::vector<Tensor> &tensor_args,
    const ast_gen_ius::TensorGroup &tensor_group,
    Expr body);

std::vector<ir::Argument> GetArgs(
    const Expr &func_body, const std::vector<std::string> &input_output_nodes);

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<ir::Argument> &args,
                                       Expr body);

std::vector<ir::Buffer> GetTempBuffers(
    const std::vector<cinn::ir::Tensor> &tensor_args, Expr body);

}  // namespace lang
}  // namespace cinn
