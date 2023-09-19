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
using poly::StageMap;

/**
 * \brief Lower the computation of \p tensor_args and \p scalar_args to a
 * LoweredFunc.
 * @param name The name of the function.
 * @param tensor_args The tensor arguments, where the computation logic locates.
 * @param scalar_args The scalar arguments, indicate some dimensions.
 * @param temp_tensors The temporary tensors(buffers) used in the body.
 * @param b The module this function belongs to.
 * @return A LoweredFunc, whose name is \p name, the argument list is the
 * concatenation of \p tensor_args and \p scalar_args.
 */
ir::LoweredFunc Lower(const std::string &name,
                      StageMap stages,
                      const std::vector<Tensor> &tensor_args,
                      const std::vector<Var> &scalar_args = {},
                      const std::vector<Tensor> &temp_tensors = {},
                      ir::Module::Builder *b = nullptr,
                      const Target &target = common::DefaultHostTarget(),
                      bool support_ir_schedule = false);

/**
 * \brief Lower the computation of \p tensor_args and \p scalar_args to a vector
 * of LoweredFuncs. Each schedule group forms a LoweredFunc.
 * @param name The name of the function.
 * @param tensor_args The tensor arguments, where the computation logic locates.
 * @param scalar_args The scalar arguments, indicate some dimensions.
 * @param temp_tensors The temporary tensors(buffers) used in the body.
 * @param b The module this function belongs to.
 * @return A vector of LoweredFuncs, whose name is \p name, name + "_1", name +
 * "_2"... The argument list is deduced from the expression of each func.
 */
std::vector<ir::LoweredFunc> LowerVec(
    const std::string &name,
    StageMap stages,
    const std::vector<Tensor> &tensor_args,
    const std::vector<Var> &scalar_args = {},
    const std::vector<Tensor> &temp_tensors = {},
    ir::Module::Builder *b = nullptr,
    const Target &target = common::DefaultHostTarget(),
    bool support_ir_schedule = false);

ir::LoweredFunc LowerToAst(const std::string &name,
                           const std::vector<Tensor> &tensor_args,
                           ast_gen_ius::TensorGroup *tensor_group,
                           const Target &target = common::DefaultHostTarget());

std::vector<ir::LoweredFunc> LowerToAstVec(
    const std::string &name,
    const std::vector<Tensor> &tensor_args,
    std::vector<ast_gen_ius::TensorGroup *> tensor_groups,
    const Target &target = common::DefaultHostTarget());

std::vector<ir::Buffer> GetTempBuffers(
    const std::vector<Tensor> &tensor_args,
    const ast_gen_ius::TensorGroup &tensor_group,
    Expr body);

std::vector<ir::Argument> GetArgs(
    const Expr &func_body, const std::vector<std::string> &input_output_nodes);

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<Tensor> &tensor_args,
                                       const poly::StageMap &stage_map,
                                       Expr body);

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<ir::Argument> &args,
                                       Expr body);

}  // namespace lang
}  // namespace cinn
