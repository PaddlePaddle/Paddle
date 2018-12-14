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

#include "paddle/fluid/inference/api/details/prepare_executor.h"
#include <string>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"

namespace paddle {
namespace inference {
namespace api {
namespace details {

static void CheckWhileOpInput(framework::ProgramDesc *program) {
  int num_while_ops = 0;
  for (size_t block_idx = 0; block_idx < program->Size(); ++block_idx) {
    framework::BlockDesc *block = program->MutableBlock(block_idx);
    for (auto *op : block->AllOps()) {
      if (op->Type() == "while") {
        VLOG(3) << "Set ExecutorPrepareContext for while_op";
        auto &op_inputs = op->Inputs();
        std::string var_name;
        if (op_inputs.find("ExecutorPrepareContext") == op_inputs.end() ||
            op->Input("ExecutorPrepareContext").size() == 0) {
          var_name = "while_prepare_context_" + std::to_string(num_while_ops++);
        } else {
          var_name = op->Input("ExecutorPrepareContext")[0];
        }
        VLOG(3) << "Name of Input(ExecutorPrepareContext): " << var_name;
        framework::VarDesc *context_var = block->Var(var_name);
        context_var->SetType(framework::proto::VarType::LOD_TENSOR);
        context_var->SetPersistable(true);

        op->SetInput("ExecutorPrepareContext", {var_name});
      }
    }
  }
  program->Flush();
}

bool PrepareExecutor(
    framework::ProgramDesc *program, framework::Executor *executor,
    framework::Scope *scope,
    std::vector<std::unique_ptr<framework::ExecutorPrepareContext>> *ctx) {
  CheckWhileOpInput(program);

  ctx->clear();
  for (size_t block_idx = 0; block_idx < program->Size(); ++block_idx) {
    ctx->push_back(executor->Prepare(*program, block_idx));
  }

  for (size_t block_idx = 0; block_idx < program->Size(); ++block_idx) {
    framework::BlockDesc *block = program->MutableBlock(block_idx);
    for (auto *op : block->AllOps()) {
      if (op->Type() == "while") {
        int id = op->GetBlockAttrId("sub_block");

        std::string var_name = op->Input("ExecutorPrepareContext")[0];
        framework::VarDesc *context_var = block->Var(var_name);

        framework::Variable *ptr = scope->Var(context_var->Name());
        framework::InitializeVariable(ptr, context_var->GetType());

        auto *tensor = ptr->GetMutable<framework::LoDTensor>();
        tensor->Resize({1});
        int64_t *data = tensor->mutable_data<int64_t>(platform::CPUPlace());
        data[0] = reinterpret_cast<int64_t>((*ctx)[id].get());
      }
    }
  }

  return true;
}

}  // namespace details
}  // namespace api
}  // namespace inference
}  // namespace paddle
