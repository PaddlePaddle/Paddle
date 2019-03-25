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

#include "paddle/fluid/imperative/tracer.h"

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

void CreateGradOp(const framework::OpDesc& op_desc,
                  const std::unordered_set<std::string>& no_grad_set,
                  const std::vector<framework::BlockDesc*>& grad_sub_block,
                  std::vector<framework::OpDesc*>* grad_op_descs,
                  std::unordered_map<std::string, std::string>* grad_to_var) {
  PADDLE_ENFORCE(grad_op_descs->empty());
  const framework::OpInfo& op_info =
      framework::OpInfoMap::Instance().Get(op_desc.Type());
  if (!op_info.grad_op_maker_) return;

  std::vector<std::unique_ptr<framework::OpDesc>> descs =
      op_info.GradOpMaker()(op_desc, no_grad_set, grad_to_var, grad_sub_block);
  for (auto& desc : descs) {
    grad_op_descs->emplace_back(desc.release());
  }
}

void InitGrad(VarBase* var, platform::DeviceContext* dev_ctx) {
  PADDLE_ENFORCE_NOT_NULL(var, "Could not get valid var base");
  PADDLE_ENFORCE_NOT_NULL(dev_ctx,
                          "Could not get valid device from forward op");

  if (var->grads_ == nullptr) {
    auto& var_t = var->var_->Get<framework::LoDTensor>();
    var->grads_ = new VarBase(var->GradName(), framework::proto::VarType::FP32,
                              framework::vectorize(var_t.dims()),
                              dev_ctx->GetPlace(), true, false);
    auto grad_t = var->grads_->var_->GetMutable<framework::LoDTensor>();
    operators::math::set_constant(*dev_ctx, grad_t, 0.0);
  }
}

platform::Place GetExpectedPlace(platform::Place place, VarBasePtrMap inputs) {
  platform::Place result = place;
  for (auto it : inputs) {
    for (VarBase* var : it.second) {
      platform::Place tmp_place =
          var->var_->Get<framework::LoDTensor>().place();
      if (!platform::is_same_place(tmp_place, result)) {
        PADDLE_THROW(
            "Input variable should keep in the same place: %s, but get place: "
            "%s of input %s instead",
            result, tmp_place, it.first);
      }
    }
  }

  return result;
}

framework::VariableNameMap CreateInputVarNameMap(
    const OpBase* op, const VarBasePtrMap& varbase_map) {
  framework::VariableNameMap result;

  auto& info_map = framework::OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(op->Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) {
    return result;
  }

  for (auto& in : op_info->Proto().inputs()) {
    auto it = varbase_map.find(in.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE(in.dispensable());
      result[in.name()] = {};
    } else {
      auto var_vector = it->second;
      std::vector<std::string> args;
      args.reserve(var_vector.size());
      for (VarBase* var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[in.name()] = args;
    }
  }
  return result;
}

framework::VariableNameMap CreateOutputVarNameMap(
    const OpBase* op, const VarBasePtrMap& varbase_map) {
  framework::VariableNameMap result;

  auto& info_map = framework::OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(op->Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) {
    return result;
  }

  for (auto& out : op_info->Proto().outputs()) {
    auto it = varbase_map.find(out.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE(out.dispensable());
      result[out.name()] = {};
    } else {
      auto var_vector = it->second;
      std::vector<std::string> args;
      args.reserve(var_vector.size());
      for (VarBase* var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[out.name()] = args;
    }
  }
  return result;
}

Tracer::Tracer(framework::BlockDesc* root_block)
    : trace_id_(0), root_block_(root_block), py_ops_() {}

void Tracer::Wait() { GetEngine()->Sync(); }

std::set<std::string> Tracer::Trace(OpBase* op, const VarBasePtrMap& inputs,
                                    VarBasePtrMap* outputs,
                                    framework::AttributeMap attrs_map,
                                    const platform::Place expected_place,
                                    const bool stop_gradient) {
  framework::VariableValueMap invars_map;
  framework::VariableValueMap outvars_map;

  // Construct input_vars_map and output_vars_map
  std::map<std::string, VarBase*> current_vars_map;
  op->input_vars_ = inputs;
  for (auto& it : op->input_vars_) {
    auto& invars = invars_map[it.first];
    invars.reserve(it.second.size());
    for (VarBase* inp : it.second) {
      PADDLE_ENFORCE_NOT_NULL(inp->var_, "op %s input %s nullptr", op->Type(),
                              inp->Name());

      invars.emplace_back(inp->var_);
      if (!stop_gradient) {
        current_vars_map[inp->Name()] = inp;
      }
      VLOG(3) << "input var name: " << inp->Name()
              << " inited: " << inp->var_->IsInitialized()
              << " stop_grad: " << inp->IsStopGradient();
    }
    op->TrackPreOp(it.first, it.second);
  }

  op->output_vars_ = *outputs;
  for (auto& it : op->output_vars_) {
    auto& outvars = outvars_map[it.first];
    const std::vector<VarBase*>& outputs = it.second;
    outvars.reserve(outputs.size());
    for (size_t i = 0U; i < outputs.size(); ++i) {
      VarBase* out = outputs[i];
      outvars.emplace_back(out->var_);
      out->TrackPreOp(op, it.first, i, stop_gradient);
      if (!stop_gradient) {
        current_vars_map[out->Name()] = out;
      }

      VLOG(3) << "input var name: " << out->Name()
              << " inited: " << out->var_->IsInitialized()
              << " stop_grad: " << out->IsStopGradient();
    }
  }

  // Check attrs and create op
  framework::VariableNameMap invars_name_map =
      CreateInputVarNameMap(op, inputs);
  framework::VariableNameMap outvars_name_map =
      CreateOutputVarNameMap(op, *outputs);

  auto& info = framework::OpInfoMap::Instance().Get(op->Type());
  if (info.Checker() != nullptr) {
    info.Checker()->Check(&attrs_map);
  }

  std::unique_ptr<framework::OperatorBase> op_base =
      framework::OpRegistry::CreateOp(op->Type(), invars_name_map,
                                      outvars_name_map, attrs_map);

  if (info.infer_var_type_) {
    RuntimeInferVarTypeContext infer_var_type_ctx(&inputs, outputs, &attrs_map);
    info.infer_var_type_(&infer_var_type_ctx);
  }

  // Run forward op
  VLOG(3) << "tracer running " << op->Type();
  framework::RuntimeContext* ctx =
      new framework::RuntimeContext(invars_map, outvars_map);

  // TODO(panyx0718): Cache p.
  framework::OperatorWithKernel* op_kernel =
      dynamic_cast<framework::OperatorWithKernel*>(op_base.get());
  op_base.release();
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");

  op->place_ = GetExpectedPlace(expected_place, inputs);

  framework::Scope scope;
  PreparedOp* prepared_op = PreparedOp::Prepare(ctx, op_kernel, op->place_);
  platform::DeviceContext* device_context = prepared_op->dev_ctx;
  prepared_op->op->RuntimeInferShape(scope, op->place_, *ctx);
  GetEngine()->Run(prepared_op);

  // construct backward op
  std::set<std::string> vars_saved_for_backward;
  if (!stop_gradient) {
    VLOG(5) << "start construct backward op";

    // construct grad op descs
    op->attrs_ = attrs_map;
    std::unique_ptr<framework::OpDesc> fwd_op_desc(new framework::OpDesc(
        op->Type(), invars_name_map, outvars_name_map, attrs_map));
    std::unique_ptr<std::unordered_map<std::string, std::string>> grad_to_var(
        new std::unordered_map<std::string, std::string>());
    // NOTE(minqiyang): We don't support control flow op in imperative now
    // Add grad_block_ when we want to support it
    CreateGradOp(*fwd_op_desc, {}, {}, &op->grad_op_descs_, grad_to_var.get());

    VLOG(5) << "create grad op desc: " << op->grad_op_descs_[0]->Type();

    const size_t grad_op_count = op->grad_op_descs_.size();

    op->grad_input_vars_.resize(grad_op_count);
    op->grad_output_vars_.resize(grad_op_count);

    for (size_t i = 0; i < grad_op_count; ++i) {
      framework::OpDesc* grad_op_desc = op->grad_op_descs_[i];
      for (auto& it : grad_op_desc->Inputs()) {
        auto& grad_in_vars = op->grad_input_vars_[i][it.first];
        grad_in_vars.reserve(it.second.size());
        for (const std::string& grad_invar : it.second) {
          const auto& var_it = grad_to_var->find(grad_invar);
          if (var_it == grad_to_var->end()) {
            auto fwd_var_it = current_vars_map.find(grad_invar);
            PADDLE_ENFORCE(fwd_var_it != current_vars_map.end());
            // Forward inputs or outputs.
            grad_in_vars.emplace_back(fwd_var_it->second);
          } else {
            VarBase* var = current_vars_map[var_it->second];
            InitGrad(var, device_context);
            // Douts.
            grad_in_vars.emplace_back(var->grads_);
          }

          vars_saved_for_backward.insert(it.first);
        }
      }

      for (auto& it : grad_op_desc->Outputs()) {
        auto& grad_out_vars = op->grad_output_vars_[i][it.first];
        for (const std::string& grad_outvar : it.second) {
          const auto& var_it = grad_to_var->find(grad_outvar);
          PADDLE_ENFORCE(var_it != grad_to_var->end(),
                         "Could not found the grad op output var, should this "
                         "operator %s's stop gradient be True",
                         op->Type());
          VarBase* var = current_vars_map[var_it->second];
          InitGrad(var, device_context);
          grad_out_vars.push_back(var->grads_);
        }
      }
    }
  }

  return vars_saved_for_backward;
}

std::vector<VarBase*> Tracer::PyTrace(OpBase* op,
                                      const std::vector<VarBase*>& inputs,
                                      bool stop_gradient) {
  VLOG(3) << "py_trace " << op->Type();

  op->input_vars_[PyLayer::kFwdInp] = inputs;

  std::vector<framework::Variable*> ret_vars =
      PyLayer::Apply(op->forward_id_, inputs);

  op->TrackPreOp(PyLayer::kFwdInp, inputs);

  std::vector<VarBase*>& outputs = op->output_vars_[PyLayer::kFwdOut];
  outputs.reserve(ret_vars.size());
  for (size_t i = 0U; i != ret_vars.size(); ++i) {
    framework::Variable* v = ret_vars[i];
    VarBase* out = new VarBase(string::Sprintf("%s_out_%d", op->Type(), i), v,
                               nullptr, stop_gradient);
    outputs.emplace_back(out);
    out->TrackPreOp(op, PyLayer::kFwdOut, i, stop_gradient);
  }

  if (!stop_gradient) {
    VLOG(5) << "start construct backward op";
    op->grad_input_vars_.resize(1);
    op->grad_output_vars_.resize(1);
    auto& grad_input_vars =
        op->grad_input_vars_[0][framework::GradVarName(PyLayer::kFwdInp)];
    auto& grad_output_vars =
        op->grad_output_vars_[0][framework::GradVarName(PyLayer::kFwdOut)];

    for (VarBase* inp : inputs) {
      grad_input_vars.push_back(inp);
    }
    for (VarBase* out : outputs) {
      grad_input_vars.push_back(out);
    }

    // TODO(minqiyang): Add GPU support for PyLayer, only support CPU now
    platform::CPUPlace place;
    for (VarBase* out : outputs) {
      InitGrad(out, platform::DeviceContextPool::Instance().Get(place));
      grad_input_vars.push_back(out->grads_);
    }

    for (VarBase* inp : inputs) {
      InitGrad(inp, platform::DeviceContextPool::Instance().Get(place));
      grad_output_vars.push_back(inp->grads_);
    }
  }
  return outputs;
}

}  // namespace imperative
}  // namespace paddle
