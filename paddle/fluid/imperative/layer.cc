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

#include "paddle/fluid/imperative/layer.h"
#include <deque>
#include <limits>
#include <map>
#include <random>
#include <utility>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace imperative {

using framework::Variable;

void AddTo(Variable* src, Variable* dst) {
  framework::LoDTensor* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  framework::LoDTensor* src_tensor = src->GetMutable<framework::LoDTensor>();
  PADDLE_ENFORCE(dst_tensor->numel() == src_tensor->numel(), "%lld vs %lld",
                 dst_tensor->numel(), src_tensor->numel());
  float* dst_data = dst_tensor->mutable_data<float>(platform::CPUPlace());
  const float* src_data = src_tensor->data<float>();
  for (size_t i = 0; i < src_tensor->numel(); ++i) {
    dst_data[i] += src_data[i];
  }
}

class Autograd {
 public:
  explicit Autograd(framework::Scope* scope) : scope_(scope) {}

  void RunBackward(VarBase* var) {
    PADDLE_ENFORCE(var->pre_op_->op_desc_);
    // TODO(panyx0718): Only create for vars that "require_grad"
    (*var->pre_op_->output_vars_)[var->pre_op_out_idx_]->grads_ = var->grads_;

    std::deque<OpBase*> ready;
    ready.push_back(var->pre_op_);

    std::map<OpBase*, int> dep_counts = ComputeDepCounts(var->pre_op_);

    while (!ready.empty()) {
      OpBase* ready_op = ready.front();
      ready.pop_front();
      std::vector<Variable*> input_grads = ready_op->ApplyGrad(scope_);

      for (size_t i = 0; i < input_grads.size(); ++i) {
        if (!input_grads[i]) continue;
        OpBase* pre_op = ready_op->pre_ops_->at(i);
        if (!pre_op) continue;

        dep_counts[pre_op] -= 1;
        PADDLE_ENFORCE(dep_counts[pre_op] >= 0);
        bool pre_op_ready = dep_counts[pre_op] == 0;
        if (pre_op_ready) {
          ready.push_back(pre_op);
        }
      }
    }
  }

 private:
  std::map<OpBase*, int> ComputeDepCounts(OpBase* op) {
    std::map<OpBase*, int> ret;

    std::deque<OpBase*> queue;
    queue.push_back(op);
    std::unordered_set<OpBase*> visited;
    visited.insert(op);
    while (!queue.empty()) {
      OpBase* candidate = queue.front();
      queue.pop_front();
      for (OpBase* pre_op : *(candidate->pre_ops_)) {
        if (!pre_op) continue;
        if (visited.find(pre_op) == visited.end()) {
          visited.insert(pre_op);
          queue.push_back(pre_op);
        }
        ret[pre_op] += 1;
      }
    }

    return ret;
  }

  framework::Scope* scope_;
};

framework::Variable* CreateVariable(const std::string& name,
                                    const framework::DDim& dim, float val,
                                    framework::Scope* scope,
                                    bool random_name = true) {
  std::string varname = name;
  if (random_name) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(
        1, std::numeric_limits<int>::max());
    int id = dist6(rng);
    varname = string::Sprintf("%s@%d", varname, id);
  }

  VLOG(3) << "creating var " << varname;
  framework::Variable* var = scope->Var(varname);
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();

  float* data = tensor->mutable_data<float>(dim, platform::CPUPlace());
  std::fill(data, data + tensor->numel(), val);
  return var;
}

framework::LoDTensor& VarBase::Grad() {
  VLOG(3) << "get var grad " << var_desc_->Name();
  return *grads_->GetMutable<framework::LoDTensor>();
}

void VarBase::ApplyGrad(framework::Scope* scope, Variable* grad) {
  VLOG(3) << "apply var grad " << var_desc_->Name() << " "
          << grad->Get<framework::LoDTensor>().data<float>()[0];
  if (!grads_) {
    grads_ =
        CreateVariable(string::Sprintf("%s@IGrad", var_desc_->Name()),
                       var_->Get<framework::LoDTensor>().dims(), 0.0, scope);
  }
  AddTo(grad, grads_);
  VLOG(3) << "grad_ after apply var grad " << var_desc_->Name() << " "
          << grads_->Get<framework::LoDTensor>().data<float>()[0];
}

std::vector<Variable*> OpBase::ApplyGrad(framework::Scope* scope) {
  VLOG(3) << "op grad " << grad_op_desc_->Type();

  for (const std::string& grad_invar : grad_op_desc_->InputArgumentNames()) {
    if (grad_to_var_->find(grad_invar) == grad_to_var_->end()) {
      // grad op inputs can be forward inputs, so not in grad_to_var.
      continue;
    }
    VLOG(3) << "op grad in var " << grad_invar;
    block_->FindRecursiveOrCreateVar(grad_invar);
    framework::Variable* var = scope->Var(grad_invar);
    const std::string& invar = grad_to_var_->at(grad_invar);
    for (VarBase* varbase : *output_vars_) {
      // Use the accumulated grads_ by sharing the input with grads_.
      if (varbase->var_desc_->Name() == invar) {
        var->GetMutable<framework::LoDTensor>()->ShareDataWith(
            varbase->grads_->Get<framework::LoDTensor>());
        break;
      }
    }
  }

  for (const std::string& outvar : grad_op_desc_->OutputArgumentNames()) {
    VLOG(3) << "grad outvar " << outvar;
    block_->FindRecursiveOrCreateVar(outvar);
    framework::Variable* var = scope->Var(outvar);
    if (!var->IsInitialized()) {
      framework::VarDesc* var_desc = block_->FindVar(outvar);
      if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
        var->GetMutable<framework::LoDTensor>();
      } else {
        LOG(ERROR) << "tracer doesn't support yet";
      }
    }
  }
  grad_op_desc_->InferShape(*block_);
  grad_op_desc_->InferVarType(block_);
  std::unique_ptr<framework::OperatorBase> opbase =
      framework::OpRegistry::CreateOp(*grad_op_desc_);

  opbase->Run(*scope, platform::CPUPlace());

  // `ret` matches exactly with `input_vars_` of forward op.
  std::vector<Variable*> ret;
  for (size_t i = 0; i < input_vars_->size(); ++i) {
    bool found = false;
    VarBase* origin_var = (*input_vars_)[i];
    for (const std::string& outvar : grad_op_desc_->OutputArgumentNames()) {
      Variable* var = scope->FindVar(outvar);
      std::string orig_var = grad_to_var_->at(outvar);
      if (origin_var->var_desc_->Name() != orig_var) {
        continue;
      }
      VLOG(3) << "apply grad " << outvar << " with origin " << orig_var;
      origin_var->ApplyGrad(scope, var);
      found = true;
      ret.push_back(var);
      // TODO(panyx0718): There might be another outvar with the same name.
      // In that case, it doesn't matter the first one or the second one is
      // used.
      break;
    }
    if (!found) {
      ret.push_back(nullptr);
    }
  }
  return ret;
}

void VarBase::RunBackward(framework::Scope* scope) {
  grads_ = CreateVariable(framework::GradVarName(var_desc_->Name()),
                          var_->Get<framework::LoDTensor>().dims(), 1.0, scope,
                          false);
  if (!pre_op_) return;
  Autograd(scope).RunBackward(this);
}

}  // namespace imperative
}  // namespace paddle
