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

static std::shared_ptr<Tracer> g_current_tracer(nullptr);

const std::shared_ptr<Tracer>& GetCurrentTracer() { return g_current_tracer; }

void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer) {
  g_current_tracer = tracer;
  VLOG(6) << "Set current tracer: " << g_current_tracer;
}

static void ClearNoNeedBufferInputs(OpBase* op) {
  auto& inferer = op->Info().NoNeedBufferVarsInferer();
  if (!inferer) return;
  auto* ins = op->GetMutableInsMap();
  const auto& no_need_buffer_slots =
      inferer(*ins, op->GetOutsMap(), op->Attrs());
  if (no_need_buffer_slots.empty()) return;

  for (auto& slot : no_need_buffer_slots) {
    auto iter = ins->find(slot);
    if (iter == ins->end()) continue;
    VLOG(2) << "Clear data buffer of " << slot << " in " << op->Type();

    for (auto& each_var : iter->second) {
      if (!each_var) continue;

      auto& var = each_var->Var();
      PADDLE_ENFORCE_EQ(var.IsType<framework::LoDTensor>(), true,
                        "Only support LoDTensor");
      // TODO(zjl): support higher order derivatives
      auto new_var = new VarBase(false, each_var->Name());
      auto* new_tensor =
          new_var->MutableVar()->GetMutable<framework::LoDTensor>();
      auto& old_tensor = var.Get<framework::LoDTensor>();
      new_tensor->Resize(old_tensor.dims());
      new_tensor->set_lod(old_tensor.lod());
      each_var.reset(new_var);
    }
  }
}

static std::vector<std::unique_ptr<OpBase>> CreateGradOpBases(
    const OpBase* fw_op_base, const NameVarBaseMap& in,
    const NameVarBaseMap& out) {
  if (fw_op_base->Info().dygraph_grad_op_maker_) {
    return fw_op_base->Info().dygraph_grad_op_maker_(fw_op_base, in, out);
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
  VLOG(1) << "Trace Op: " << type;
  size_t op_id = GenerateUniqueId();
  auto op = OpBase::Create(op_id, type, ins, outs, attrs, place);
  op->Run(ins, outs);

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, ins, outs, op->Attrs());
  }

  if (ComputeRequiredGrad(ins, outs, trace_backward)) {
    TraceBackward(op, ins, outs);
  } else {
    VLOG(3) << "No Grad to track for Op: " << type;
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs,
                     framework::AttributeMap attrs) {
  VLOG(1) << "Trace Op: " << type;
  size_t op_id = GenerateUniqueId();
  auto op =
      OpBase::Create(op_id, type, ins, outs, std::move(attrs), expected_place_);
  op->Run(ins, outs);

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, ins, outs, op->Attrs());
  }

  if (ComputeRequiredGrad(ins, outs, no_grad_)) {
    TraceBackward(op, ins, outs);
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
                           const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs) {
  // grad_to_var is a map of framework::GradVarName(in_var_name/out_var_name) ->
  // in_var_name/out_var_name
  std::unordered_map<std::string, std::string> grad_to_var;

  // Get grad_op_desc using fwd_op_desc
  std::vector<std::unique_ptr<OpBase>> grad_op_bases_ =
      CreateGradOpBases(fwd_op.get(), ins, outs);

  size_t grad_op_num = grad_op_bases_.size();

  std::set<VarBase*> set_input_vars;
  for (auto& fwd_in_it : ins) {
    for (auto& var_base_it : fwd_in_it.second) {
      set_input_vars.insert(var_base_it.get());
    }
  }

  for (auto& fwd_out_it : outs) {
    for (auto& var_base_it : fwd_out_it.second) {
      set_input_vars.insert(var_base_it.get());
    }
  }

  for (size_t i = 0; i < grad_op_num; ++i) {
    size_t trace_id = fwd_op->id();

    std::shared_ptr<OpBase> grad_op = std::move(grad_op_bases_[i]);
    grad_op->SetId(trace_id);
    grad_op->SetPlace(fwd_op->place());
    grad_op->CreateOperatorBase();

    auto& grad_in = *(grad_op->GetMutableInsMap());
    auto& grad_out = *(grad_op->GetMutableOutsMap());
    for (auto& grad_in_it : grad_in) {
      for (auto& var_base_it : grad_in_it.second) {
        if (set_input_vars.count(var_base_it.get()) == 0) {
          var_base_it->AddGradOps(grad_op);
          engine_->InsertGradVar(var_base_it.get());
        }
      }
    }

    std::set<OpBase*> visited_preceding_ops;
    for (auto& grad_out_it : grad_out) {
      bool flag_clear_list = false;
      for (auto& var_base_it : grad_out_it.second) {
        if ((!var_base_it->OverridedStopGradient()) ||
            (grad_out_it.second.size() > 1)) {
          auto preceding_ops = var_base_it->GradOps();
          if (!preceding_ops.empty()) {
            for (const auto& op : preceding_ops) {
              visited_preceding_ops.insert(op);
            }
          }
        } else {
          flag_clear_list = true;
        }
      }
      if (flag_clear_list) {
        grad_out_it.second.clear();
      }
    }
    std::vector<OpBase*> vec_preceding_ops(visited_preceding_ops.begin(),
                                           visited_preceding_ops.end());

    grad_op->SetGradPendingOps(std::move(vec_preceding_ops));

    // this OpBase* is just used to manage op's life time
    engine_->InsertOp(grad_op.get(), grad_op);
    ClearNoNeedBufferInputs(grad_op.get());
  }
}

}  // namespace imperative
}  // namespace paddle
