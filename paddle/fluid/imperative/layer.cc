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
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace imperative {

using framework::Variable;

void AddTo(Variable* src, Variable* dst) {
  framework::LoDTensor* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  framework::LoDTensor* src_tensor = src->GetMutable<framework::LoDTensor>();
  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (src_tensor->numel() == 0) {
    return;
  }
  PADDLE_ENFORCE(dst_tensor->numel() == src_tensor->numel(),
                 "dst_numel %lld vs. src_numel %lld", dst_tensor->numel(),
                 src_tensor->numel());
  float* dst_data = dst_tensor->mutable_data<float>(platform::CPUPlace());
  const float* src_data = src_tensor->data<float>();
  for (size_t i = 0; i < src_tensor->numel(); ++i) {
    dst_data[i] += src_data[i];
  }
}

class Autograd {
 public:
  Autograd() {}

  void RunBackward(VarBase* var) {
    if (var->stop_gradient_) {
      return;
    }

    std::deque<OpBase*> ready;
    ready.push_back(var->pre_op_);

    std::map<OpBase*, int> dep_counts = ComputeDepCounts(var->pre_op_);

    while (!ready.empty()) {
      OpBase* ready_op = ready.front();
      ready.pop_front();
      std::map<std::string, std::vector<VarBase*>> input_grads =
          ready_op->ApplyGrad();

      for (auto it : input_grads) {
        const std::vector<VarBase*>& ingrads = it.second;
        for (size_t i = 0; i < ingrads.size(); ++i) {
          if (!ingrads[i]) continue;
          if (ready_op->input_vars_[it.first][i]->stop_gradient_) {
            continue;
          }
          OpBase* pre_op = ready_op->pre_ops_[it.first][i];
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
      for (auto it : candidate->pre_ops_) {
        for (OpBase* pre_op : it.second) {
          if (!pre_op) continue;
          if (visited.find(pre_op) == visited.end()) {
            visited.insert(pre_op);
            queue.push_back(pre_op);
          }
          ret[pre_op] += 1;
        }
      }
    }
    return ret;
  }
};

framework::LoDTensor& VarBase::Grad() {
  VLOG(3) << "get var grad " << var_desc_->Name();
  return *grads_->GetMutable<framework::LoDTensor>();
}

std::map<std::string, std::vector<VarBase*>> OpBase::ApplyGrad() {
  if (!grad_op_desc_) {
    LOG(WARNING) << "op with no grad: " << op_desc_->Type();
    return {};
  }
  VLOG(3) << "op grad " << grad_op_desc_->Type();

  std::vector<std::unique_ptr<framework::Variable>> tmp_vars;
  std::map<std::string, std::vector<framework::Variable*>> grad_outputs;
  for (auto it : grad_output_vars_) {
    auto& outputs = grad_outputs[it.first];
    for (size_t i = 0; i < it.second.size(); ++i) {
      // Allocate a new variable
      Variable* tmp_var = new framework::Variable();
      tmp_var->GetMutable<framework::LoDTensor>();

      tmp_vars.emplace_back(tmp_var);
      outputs.push_back(tmp_var);
    }
  }

  framework::RuntimeContext ctx(grad_input_vars_, grad_outputs);

  // No need to do compile time infer shape here.
  // grad_op_desc_->InferShape(*block_);
  grad_op_desc_->InferVarType(block_);

  std::unique_ptr<framework::OperatorBase> opbase =
      framework::OpRegistry::CreateOp(*grad_op_desc_);
  framework::OperatorWithKernel* op_kernel =
      dynamic_cast<framework::OperatorWithKernel*>(opbase.get());
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");

  framework::Scope scope;
  platform::CPUPlace place;
  PreparedOp p = PreparedOp::Prepare(ctx, *op_kernel, place);
  p.op.RuntimeInferShape(scope, place, ctx);
  p.func(framework::ExecutionContext(p.op, scope, *p.dev_ctx, p.ctx));

  for (auto it : grad_output_vars_) {
    auto& outputs = grad_outputs[it.first];
    auto& origin_outputs = it.second;

    for (size_t i = 0; i < outputs.size(); ++i) {
      framework::Variable* orig_grad = origin_outputs[i];
      AddTo(outputs[i], orig_grad);
    }
  }
  return input_vars_;
}

void VarBase::RunBackward() {
  if (!pre_op_) return;

  auto grads_t = grads_->GetMutable<framework::LoDTensor>();
  float* data = grads_t->mutable_data<float>(platform::CPUPlace());
  std::fill(data, data + grads_t->numel(), 1.0);

  PADDLE_ENFORCE(
      grads_ ==
      pre_op_->output_vars_[pre_op_out_name_][pre_op_out_idx_]->grads_);
  Autograd().RunBackward(this);
}

}  // namespace imperative
}  // namespace paddle
