/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/passes/ir_data_transform_pass.h"
#include <vector>
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace inference {
namespace analysis {

bool IrDataTransformPass::IsSpecialOp(
    const std::vector<std::string> &special_ops, std::string type) const {
  for (auto name : special_ops) {
    if (name == type) {
      return true;
    }
  }
  return false;
}

bool IrDataTransformPass::NeedTransform(const framework::BlockDesc *block,
                                        const framework::OpDesc *op,
                                        const std::string &var_name,
                                        size_t index) const {
  // For some op, the input can be either on CPU or GPU, and the output will be
  // on the GPU (for CUDAPlace).
  std::vector<std::string> special_gpu_ops = {"read_from_array",
                                              "write_to_array"};
  // For some op, the input can be either on CPU or GPU, and the output will be
  // on the CPU.
  std::vector<std::string> special_cpu_ops = {"is_empty"};
  // LogicalOp kernel's device type is decided by input tensor place
  std::vector<std::string> special_ops = {"logical_and",    "logical_or",
                                          "logical_not",    "logical_xor",
                                          "data_transform", "while"};

  if (IsSpecialOp(special_ops, op->Type()) ||
      IsSpecialOp(special_gpu_ops, op->Type()) ||
      IsSpecialOp(special_cpu_ops, op->Type())) {
    return false;
  }

  framework::OpDesc *previous_op = nullptr;
  std::vector<framework::OpDesc *> all_ops = block->AllOps();
  for (int i = index - 1; i >= 0; --i) {
    for (auto output_var_name : all_ops[i]->OutputArgumentNames()) {
      if (output_var_name == var_name) {
        previous_op = all_ops[i];
        break;
      }
    }
    if (previous_op) {
      break;
    }
  }
  if (previous_op) {
    if (IsSpecialOp(special_ops, previous_op->Type())) {
      return false;
    }
    if (framework::OpSupportGPU(previous_op->Type()) &&
        !framework::OpSupportGPU(op->Type())) {
      VLOG(3) << "var_name: " << var_name
              << ", previous_op: " << previous_op->Type() << "[GPU]"
              << ", op: " << op->Type() << "[CPU]";
      return true;
    }
    if (!framework::OpSupportGPU(previous_op->Type()) &&
        framework::OpSupportGPU(op->Type())) {
      VLOG(3) << "var_name: " << var_name
              << ", previous_op: " << previous_op->Type() << "[CPU]"
              << ", op: " << op->Type() << "[GPU]";
      // TODO(liuyiqun): data_transform_op does not support this yet.
      return false;
    }
  }
  return false;
}

void IrDataTransformPass::InsertDataTransformOp(framework::BlockDesc *block,
                                                framework::OpDesc *op,
                                                const std::string &input_name,
                                                const std::string &var_name,
                                                size_t index) {
  // New transformed variables
  std::string transformed_var_name = "transformed_" + var_name;
  framework::VarDesc *transformed_var = block->Var(transformed_var_name);
  transformed_var->SetType(framework::proto::VarType::LOD_TENSOR);
  transformed_var->SetPersistable(false);

  // Add data_transform op
  framework::OpDesc *transform_op = block->InsertOp(index);
  transform_op->SetType("data_transform");
  transform_op->SetInput("X", {var_name});
  transform_op->SetOutput("Out", {transformed_var_name});

  // Update the input of beam_search op
  op->RenameInput(var_name, transformed_var_name);
}

void IrDataTransformPass::RunImpl(Argument *argument) {
  LOG(WARNING) << "This pass is used to add a data_transform op to transfer "
                  "data from GPU to CPU.";

  ARGUMENT_CHECK_FIELD(argument, ir_analyzed_program);
  PADDLE_ENFORCE(argument->use_gpu_valid());

  if (!argument->use_gpu()) return;

  framework::ProgramDesc program(argument->ir_analyzed_program());
  for (size_t block_idx = 0; block_idx < program.Size(); ++block_idx) {
    framework::BlockDesc *block = program.MutableBlock(block_idx);

    for (size_t op_index = 0; op_index < block->OpSize(); ++op_index) {
      framework::OpDesc *op = block->Op(op_index);

      for (std::string name : op->InputNames()) {
        for (std::string var_name : op->Input(name)) {
          if (NeedTransform(block, op, var_name, op_index)) {
            InsertDataTransformOp(block, op, name, var_name, op_index);
            op_index++;
          }
        }
      }
    }
  }
  program.Flush();
  argument->SetIrAnalyzedProgram(
      new framework::proto::ProgramDesc(*program.Proto()));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
