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
#include <unordered_set>
#include <utility>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

static std::vector<std::unique_ptr<framework::OpDesc>> CreateGradOpDescs(
    const framework::OpInfo& op_info, const framework::OpDesc& op_desc,
    const std::unordered_set<std::string>& no_grad_set,
    const std::vector<framework::BlockDesc*>& grad_sub_block,
    std::unordered_map<std::string, std::string>* grad_to_var) {
  if (op_info.grad_op_maker_) {
    return op_info.grad_op_maker_(op_desc, no_grad_set, grad_to_var,
                                  grad_sub_block);
  } else {
    return {};
  }
}

static void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& name_pair : outs) {
    for (const auto& vb : name_pair.second) {
      VLOG(6) << "Set output: " << vb->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      vb->InnerSetOverridedStopGradient(generate_grad);
    }
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
    TraceBackward(op, framework::OpDesc(op->Type(), op->InputNameMap(),
                                        op->OutputNameMap(), op->Attrs()),
                  ins, outs);
  } else {
    VLOG(3) << "No Grad to track for Op: " << type;
  }
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap& outs,
                                 bool trace_backward) {
  if (!trace_backward) return false;

  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverridedStopGradient());
        return true;
      }
    }
  }
  return false;
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
      CreateGradOpDescs(fwd_op->Info(), fwd_op_desc, {}, {}, &grad_to_var);

  // Create grad_ops using grad_op_descs

  size_t grad_op_num = grad_op_descs_.size();

  VLOG(3) << "Create " << grad_op_num << " grad op desc(s) to op "
          << fwd_op->Type();

  if (grad_op_num == 0) {
    return;
  }
  // Build a map to record var_name -> std::shared_ptr<VarBase>*,
  // so that we can find suitable var in grad op descs
  std::unordered_map<std::string, const std::shared_ptr<VarBase>*> name_to_var;
  for (auto& pair : ins) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE_EQ(var_ptr == nullptr || var_ptr->get() == var.get(), true,
                        "There are different variables with same name %s",
                        var->Name());
      var_ptr = &var;
    }
  }

  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE_EQ(var_ptr == nullptr || var_ptr->get() == var.get(), true,
                        "There are different variables with same name %s",
                        var->Name());
      var_ptr = &var;
    }
  }

  // Build backward ins and outs

  for (size_t i = 0; i < grad_op_num; i++) {
    // Step1: build grad op and add them to engine

    // Use trace id to decide the order of gradient sum in sorted sum mode
    size_t trace_id = fwd_op->id();
    std::shared_ptr<OpBase> grad_op =
        OpBase::Create(trace_id, (*(grad_op_descs_[i].get())), fwd_op->place());

    // this OpBase* is just used to manage op's life time
    engine_->InsertOp(grad_op.get(), grad_op);

    std::unordered_set<OpBase*> visited_preceding_ops;
    // Step2 : prepare grad_in vars and bind them with grad_op,
    // set inputs' grad_op as current grad_op
    for (const auto& grad_ins : grad_op_descs_[i]->Inputs()) {
      if (grad_ins.second.empty()) continue;
      auto& bwd_in = (*grad_op->GetMutableInsMap())[grad_ins.first];
      bwd_in.reserve(grad_ins.second.size());

      for (auto& grad_in_var_name : grad_ins.second) {
        auto iter = grad_to_var.find(grad_in_var_name);

        if (iter != grad_to_var.end()) {
          // If it is a grad var, find its coresponding forward var
          auto& fwd_var_name = iter->second;
          auto fwd_var_iter = name_to_var.find(fwd_var_name);
          PADDLE_ENFORCE_EQ(fwd_var_iter != name_to_var.end(), true,
                            "Cannot find forward variable named %s",
                            fwd_var_name);
          const auto& tmp = (*(fwd_var_iter->second))->GradVarBase();
          PADDLE_ENFORCE_NOT_NULL(
              tmp.get(),
              "Grad of %s should "
              "not be NULL when we Track_Backward Input of %s",
              (*(fwd_var_iter->second))->Name(), grad_op->Type());
          // Create grad_in's dim in tensor for Grad Dependency compute
          auto* tensor = tmp->MutableVar()->GetMutable<framework::LoDTensor>();
          tensor->Resize((*(fwd_var_iter->second))
                             ->Var()
                             .Get<framework::LoDTensor>()
                             .dims());
          // Add Grad Op for grad_in
          tmp->AddGradOps(grad_op);
          VLOG(3) << "Add Grad Op " << grad_op->Type() << " for :"
                  << (*(fwd_var_iter->second))->GradVarBase()->Name();
          // Add Grad var input to engine set
          engine_->InsertGradVar(tmp.get());
          VLOG(3) << "Add Grad: " << tmp->Name() << " in to Engine";
          bwd_in.emplace_back((*(fwd_var_iter->second))->GradVarBase());
        } else {
          // If it is a forward var, just add it
          auto fwd_var_iter = name_to_var.find(grad_in_var_name);
          PADDLE_ENFORCE_EQ(fwd_var_iter != name_to_var.end(), true,
                            "Cannot find forward variable named %s",
                            grad_in_var_name);
          bwd_in.emplace_back(*(fwd_var_iter->second));
        }
        VLOG(3) << "Set backward input from fwd var" << grad_ins.first << " of "
                << grad_op->Type() << " to be "
                << (bwd_in.back() ? bwd_in.back()->Name() : "nullptr");
      }
    }

    // Step3: prepare grad_out vars and using their grad_ops to set current
    // grad_op's preceding op
    for (auto& grad_outs : grad_op_descs_[i]->Outputs()) {
      if (grad_outs.second.empty()) continue;
      auto& bwd_out = (*grad_op->GetMutableOutsMap())[grad_outs.first];
      bwd_out.reserve(grad_outs.second.size());

      for (auto& grad_out_var_name : grad_outs.second) {
        auto iter = grad_to_var.find(grad_out_var_name);
        PADDLE_ENFORCE_EQ(iter != grad_to_var.end(), true,
                          "Cannot find output of input grad %s in op %s",
                          grad_out_var_name, fwd_op->Type());
        auto fwd_var_iter = name_to_var.find(iter->second);
        PADDLE_ENFORCE_EQ(fwd_var_iter != name_to_var.end(), true,
                          "Cannot find forward variable named %s",
                          iter->second);
        const auto& tmp = (*(fwd_var_iter->second))->GradVarBase();

        PADDLE_ENFORCE_NOT_NULL(tmp.get(),
                                "Grad output: %s of op: %s should not be NULL",
                                (tmp->Name(), grad_op->Type()));

        if ((!tmp->OverridedStopGradient()) || (grad_outs.second.size() > 1)) {
          VLOG(3) << "Set backward output " << grad_outs.first << " of "
                  << grad_op->Type() << " to be " << tmp->Name()
                  << ". Its Overrided Stop_Gradient is: False";
          bwd_out.emplace_back(tmp);
          auto grad_pending_ops =
              (*(fwd_var_iter->second))->GradVarBase()->GradOps();
          if (VLOG_IS_ON(3) && !grad_pending_ops.empty()) {
            VLOG(3) << "Add grad_pending Op of :"
                    << (*(fwd_var_iter->second))->GradVarBase()->Name()
                    << " It's grad_pending Op are: ";
            for (const auto& op : grad_pending_ops) {
              VLOG(3) << op->Type();
            }
          }
          if (!grad_pending_ops.empty()) {
            for (const auto& op : grad_pending_ops) {
              PADDLE_ENFORCE_NOT_NULL(op,
                                      "No nullptr should be grad_pending op");
              if (visited_preceding_ops.count(op) == 0) {
                visited_preceding_ops.insert(op);
                grad_op->InsertGradPendingOps(op);
              }
            }
          } else {
            VLOG(5) << "Hit leaf VarBase"
                    << (*(fwd_var_iter->second))->GradVarBase()->Name();
          }
        } else {
          VLOG(3) << "Skip backward output " << grad_outs.first << " of "
                  << grad_op->Type() << " Named: " << tmp->Name()
                  << ", since its Overrided Stop_Gradient is: True";
        }
      }
    }
    // To ensure numeric stability as static graph
    grad_op->SortGradPendingOps();
  }
}

}  // namespace imperative
}  // namespace paddle
