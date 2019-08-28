// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/imperative/tracer.h"
#include <set>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

struct OpBaseCmp {
  bool operator()(OpBase* first, OpBase* second) {
    return first->id() > second->id();
  }
};

static std::vector<std::unique_ptr<framework::OpDesc>> CreateGradOpDescs(
    const framework::OpInfo& op_info, const framework::OpDesc& op_desc,
    const std::unordered_set<std::string>& no_grad_set,
    const std::vector<framework::BlockDesc*>& grad_sub_block,
    std::unordered_map<std::string, std::string>* grad_to_var,
    const NameVarBaseMap* in, const NameVarBaseMap* out) {
  if (op_info.grad_op_maker_) {
    return op_info.grad_op_maker_(op_desc, no_grad_set, grad_to_var,
                                  grad_sub_block, in, out);
  } else {
    return {};
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward) {
  platform::RecordEvent event(type);
  VLOG(1) << "Trace Op: " << type;
  size_t op_id = GenerateUniqueId();
  auto op = OpBase::Create(op_id, type, ins, outs, std::move(attrs), place);
  op->Run(ins, outs);

  if (ComputeRequiredGrad(ins, outs, trace_backward)) {
    auto fw_op_desc = framework::OpDesc();
    fw_op_desc.SetType(type);
    TraceBackward(op, fw_op_desc, ins, outs);
  }
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap outs,
                                 bool trace_backward) {
  // TODO(jiabin): Implement auto prune here
  return trace_backward;
}

void Tracer::TraceBackward(const std::shared_ptr<OpBase>& fwd_op,
                           const framework::OpDesc& fwd_op_desc,
                           const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs) {
  // grad_to_var is a map of framework::GradVarName(in_var_name/out_var_name) ->
  // in_var_name/out_var_name
  std::unordered_map<std::string, std::string> grad_to_var;

  // Get grad_op_desc using fwd_op_desc
  std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs_ =
      CreateGradOpDescs(fwd_op->Info(), fwd_op_desc, {}, {}, &grad_to_var, &ins,
                        &outs);

  size_t grad_op_num = grad_op_descs_.size();

  for (size_t i = 0; i < grad_op_num; ++i) {
    size_t trace_id = fwd_op->id();

    auto& temp_in = grad_op_descs_[i]->DygraphInput();
    auto& temp_out = grad_op_descs_[i]->DygraphOutput();
    std::shared_ptr<OpBase> grad_op =
        OpBase::Create(trace_id, grad_op_descs_[i]->Type(), temp_in, temp_out,
                       fwd_op->Attrs(), fwd_op->place());

    grad_op_descs_[i]->GetDygraphInput(grad_op->GetMutableInsMap());
    grad_op_descs_[i]->GetDygraphOutput(grad_op->GetMutableOutsMap());

    auto& grad_in = *(grad_op->GetMutableInsMap());
    auto& grad_out = *(grad_op->GetMutableOutsMap());
    for (auto& grad_in_it : grad_in) {
      for (auto& var_base_it : grad_in_it.second) {
        if (var_base_it->IsGradFromGradMaker() == true) {
          var_base_it->AddGradOps(grad_op);
        }
      }
    }
    std::set<OpBase*, OpBaseCmp> visited_preceding_ops;
    for (auto& grad_out_it : grad_out) {
      for (auto& var_base_it : grad_out_it.second) {
        auto preceding_ops = var_base_it->GradOps();

        if (!preceding_ops.empty()) {
          for (const auto& op : preceding_ops) {
            visited_preceding_ops.insert(op);
          }
        }
      }
    }
    std::vector<OpBase*> vec_preceding_ops(visited_preceding_ops.begin(),
                                           visited_preceding_ops.end());

    grad_op->SetPrecedingOps(vec_preceding_ops);

    // this OpBase* is just used to manage op's life time
    engine->InsertOp(grad_op.get(), grad_op);
  }
}

}  // namespace imperative
}  // namespace paddle
