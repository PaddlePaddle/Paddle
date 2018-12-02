// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace imperative {

void CreateGradOp(const framework::OpDesc& op_desc,
                  const std::unordered_set<std::string>& no_grad_set,
                  const std::vector<framework::BlockDesc*>& grad_sub_block,
                  framework::OpDesc** grad_op_desc,
                  std::unordered_map<std::string, std::string>* grad_to_var) {
  std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs =
      framework::OpInfoMap::Instance()
          .Get(op_desc.Type())
          .GradOpMaker()(op_desc, no_grad_set, grad_to_var, grad_sub_block);
  PADDLE_ENFORCE(grad_op_descs.size() == 1, "Only support 1 grad op now.");
  // TODO(panyx0718): Leak?
  *grad_op_desc = grad_op_descs[0].release();
}

class Tracer {
 public:
  Tracer() {}

  void Trace(OpBase* op, const std::vector<VarBase*>& inputs,
             const std::vector<VarBase*>& outputs) {
    framework::OpDesc* op_desc = op->op_desc_;
    LOG(ERROR) << "tracer tracing " << op_desc->Type();
    op_desc->InferShape(*block_);
    op_desc->InferVarType(block_);
    std::unique_ptr<framework::OperatorBase> op_base =
        framework::OpRegistry::CreateOp(*op_desc);

    *op->input_vars_ = inputs;
    for (VarBase* input : inputs) {
      const std::string vname = input->var_desc_->Name();
      framework::Variable* var = scope_->Var(vname);
      input->var_ = var;
      if (!var->IsInitialized()) {
        framework::VarDesc* var_desc = block_->FindVar(vname);
        if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
          var->GetMutable<framework::LoDTensor>();
        } else {
          LOG(ERROR) << "tracer doesn't support yet";
        }
      }
      if (input->pre_op_) {
        op->pre_ops_->push_back(input->pre_op_);
        op->pre_ops_out_idx_->push_back(input->pre_op_out_idx_);
      } else {
        op->pre_ops_->push_back(nullptr);
      }
    }

    *op->output_vars_ = outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string vname = outputs[i]->var_desc_->Name();
      framework::Variable* var = scope_->Var(vname);
      if (!var->IsInitialized()) {
        framework::VarDesc* var_desc = block_->FindVar(vname);
        if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
          var->GetMutable<framework::LoDTensor>();
        } else {
          LOG(ERROR) << "tracer doesn't support yet";
        }
      }
      outputs[i]->var_ = var;
      outputs[i]->pre_op_ = op;
      outputs[i]->pre_op_out_idx_ = i;
    }
    op_base->Run(*scope_, platform::CPUPlace());

    framework::OpDesc* grad_op_desc;
    auto grad_to_var = new std::unordered_map<std::string, std::string>();
    CreateGradOp(*op_desc, {}, {block_}, &grad_op_desc, grad_to_var);
    op->grad_op_desc_ = grad_op_desc;
    op->grad_to_var_ = grad_to_var;
    op->block_ = block_;
  }

  void SetScope(framework::Scope* scope) { scope_ = scope; }

  void SetBlock(framework::BlockDesc* block) { block_ = block; }

  framework::Scope* Scope() const { return scope_; }

  framework::BlockDesc* Block() const { return block_; }

 private:
  framework::BlockDesc* block_;
  framework::Scope* scope_;
  std::vector<Runnable*> runnables_;
};

}  // namespace imperative
}  // namespace paddle
