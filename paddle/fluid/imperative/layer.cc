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
  Autograd() {}

  void RunBackward(VarBase* var) {
    PADDLE_ENFORCE(var->pre_op_->op_desc_);
    PADDLE_ENFORCE(
        var->grads_ ==
        var->pre_op_->output_vars_[var->pre_op_out_name_][var->pre_op_out_idx_]
            ->grads_);

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
          OpBase* pre_op = (*ready_op->pre_ops_)[it.first][i];
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
      for (auto it : *(candidate->pre_ops_)) {
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

void CreateVariable(const std::string& name, const framework::DDim& dim,
                    float val, bool random_name, framework::Variable* var) {
  if (var->IsInitialized()) return;

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
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
  float* data = tensor->mutable_data<float>(dim, platform::CPUPlace());
  std::fill(data, data + tensor->numel(), val);
}

framework::LoDTensor& VarBase::Grad() {
  VLOG(3) << "get var grad " << var_desc_->Name();
  return *grads_->GetMutable<framework::LoDTensor>();
}

std::map<std::string, std::vector<VarBase*>> OpBase::ApplyGrad() {
  if (!grad_op_desc_) {
    VLOG(3) << "op with no grad: " << op_desc_->Type();
    return {};
  }
  VLOG(3) << "op grad " << grad_op_desc_->Type();

  std::map<std::string, std::vector<framework::Variable*>> grad_outputs;
  for (auto it : grad_output_vars_) {
    auto& outputs = grad_outputs[it.first];
    for (size_t i = 0; i < it.second.size(); ++i) {
      outputs.push_back(new framework::Variable());
      outputs.back()->GetMutable<framework::LoDTensor>();
    }
  }

  framework::RuntimeContext ctx(grad_input_vars_, grad_outputs);

  // No need to do static infer shape here.
  // grad_op_desc_->InferShape(*block_);
  grad_op_desc_->InferVarType(block_);

  std::unique_ptr<framework::OperatorBase> opbase =
      framework::OpRegistry::CreateOp(*grad_op_desc_);
  opbase->Run(ctx, platform::CPUPlace());

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
  auto grads_t = grads_->GetMutable<framework::LoDTensor>();
  float* data = grads_t->mutable_data<float>(platform::CPUPlace());
  std::fill(data, data + grads_t->numel(), 1.0);

  if (!pre_op_) return;
  Autograd().RunBackward(this);
}

}  // namespace imperative
}  // namespace paddle
