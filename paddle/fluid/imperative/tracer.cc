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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

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
                        platform::errors::InvalidArgument(
                            "The type of Variable var should be LoDTensor."));
      // TODO(zjl): support higher order derivatives
      auto new_var = new VariableWrapper(each_var->Name());
      auto* new_tensor =
          new_var->MutableVar()->GetMutable<framework::LoDTensor>();
      auto& old_tensor = var.Get<framework::LoDTensor>();
      new_tensor->Resize(old_tensor.dims());
      new_tensor->set_lod(old_tensor.lod());
      each_var.reset(new_var);
      op->AddAllowedEmptyVar(new_var);
    }
  }
}

static std::vector<std::shared_ptr<OpBase>> CreateGradOpBases(
    const framework::OpInfo& info, const std::string& type,
    const NameVarBaseMap& in, const NameVarBaseMap& out,
    const framework::AttributeMap& attrs) {
  if (info.dygraph_grad_op_maker_) {
    return info.dygraph_grad_op_maker_(type, in, out, attrs);
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
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true);
  }

  OpBase::Run(*op, ins, outs, attrs, place);

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, ins, outs, attrs);
  }

  if (ComputeRequiredGrad(ins, outs, trace_backward)) {
    TraceBackward(op_info, type, ins, outs, attrs, place);
  } else {
    VLOG(3) << "No Grad to track for Op: " << type;
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs,
                     framework::AttributeMap attrs) {
  TraceOp(type, ins, outs, std::move(attrs), expected_place_, no_grad_);
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

void Tracer::TraceBackward(const framework::OpInfo& info,
                           const std::string& type, const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs,
                           const framework::AttributeMap& attrs,
                           const platform::Place& place) {
  auto grad_op_bases = CreateGradOpBases(info, type, ins, outs, attrs);
  auto grad_op_num = grad_op_bases.size();
  if (grad_op_num == 0) return;

  size_t trace_id = GenerateUniqueId();
  for (auto& grad_op : grad_op_bases) {
    grad_op->SetPlace(place);
    grad_op->SetId(trace_id);
    ClearNoNeedBufferInputs(grad_op.get());
  }
}

}  // namespace imperative
}  // namespace paddle
