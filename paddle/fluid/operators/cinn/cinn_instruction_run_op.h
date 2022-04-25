// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/instruction.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"
#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include "paddle/fluid/operators/cinn/cinn_op_helper.h"

namespace paddle::operators {

using CinnInstruction = ::cinn::hlir::framework::Instruction;
using CinnCompiledObject = framework::paddle2cinn::CinnCompiledObject;
using CinnCompiler = framework::paddle2cinn::CinnCompiler;

template <typename DeviceContext, typename T>
class CinnInstructionRunOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // step 1: fetch the cinn instruction bound to this operator
    auto cached_index = ctx.template Attr<int64_t>(kCachedIndex);
    auto ins_index = ctx.template Attr<int64_t>(kInstructionIndex);
    const CinnCompiledObject& compiled_object =
        CinnCompiler::GetInstance()->GetCompiledObject(cached_index);
    const std::vector<std::unique_ptr<CinnInstruction>>& instructions =
        compiled_object.runtime_program->GetRunInstructions();
    PADDLE_ENFORCE_LT(ins_index, instructions.size(),
                      platform::errors::InvalidArgument(
                          "Index(%ld) > instructions.size(%ld).", ins_index,
                          instructions.size()));
    auto&& instruction = instructions.at(ins_index);

    // step 2: prepare the input and output arguments of the instruction
    details::CinnLaunchContext* launch_context =
        compiled_object.launch_context.get();
    auto share_argument_buffer_fn = [launch_context,
                                     &ctx](const std::string& var_name) {
      cinn_buffer_t* buffer = launch_context->GetCinnBufferOfVar(var_name);
      framework::Variable* var = ctx.scope().GetVar(var_name);
      auto* tensor = var->template GetMutable<framework::LoDTensor>();
      buffer->memory = reinterpret_cast<uint8_t*>(tensor->mutable_data(
          ctx.GetPlace(),
          framework::paddle2cinn::TransToPaddleDataType(buffer->type)));
    };
    std::vector<std::string> in_args = ctx.InputNames(kX);
    std::for_each(in_args.begin(), in_args.end(), share_argument_buffer_fn);
    std::vector<std::string> out_args = ctx.OutputNames(kOutputs);
    std::for_each(out_args.begin(), out_args.end(), share_argument_buffer_fn);

    // step 3: launch CINN runtime to execute the instruction
    // TODO(CtfGo): simplify format of arguments package as a vector in CINN
    // and update this usage call
    instruction->Run(&launch_context->FinalizeArguments(), false,
                     details::GetStream<DeviceContext>(ctx));
  }
};

}  // namespace paddle::operators
