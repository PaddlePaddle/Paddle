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
#include <utility>

#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

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

void CreateNoBuffuerGrad(std::shared_ptr<imperative::VarBase> var,
                         platform::DeviceContext* dev_ctx) {
  PADDLE_ENFORCE_NOT_NULL(var, "Could not get valid var base");
  PADDLE_ENFORCE_NOT_NULL(dev_ctx,
                          "Could not get valid device from forward op");

  if (var->grads_ == nullptr) {
    auto& var_t = var->var_->Get<framework::LoDTensor>();
    var->grads_ = std::shared_ptr<imperative::VarBase>(
        new VarBase(var->GradName(), framework::proto::VarType::FP32,
                    framework::vectorize(var_t.dims()), dev_ctx->GetPlace(),
                    var->IsStopGradient(), false, false));
  }
}

platform::Place GetExpectedPlace(platform::Place place, VarBasePtrMap inputs) {
  platform::Place result = place;
  for (const auto& it : inputs) {
    for (const std::shared_ptr<imperative::VarBase>& var : it.second) {
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
      for (std::shared_ptr<imperative::VarBase> var_base : var_vector) {
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
      for (const std::shared_ptr<imperative::VarBase>& var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[out.name()] = args;
    }
  }
  return result;
}

Tracer::Tracer(framework::BlockDesc* root_block) : root_block_(root_block) {}

void Tracer::Trace(OpBase* op, const VarBasePtrMap& inputs,
                   VarBasePtrMap* outputs, framework::AttributeMap attrs_map,
                   const platform::Place expected_place,
                   const bool stop_gradient) {
  platform::RecordEvent record_event(op->type_);
  framework::VariableValueMap invars_map;
  framework::VariableValueMap outvars_map;

  // Construct input_vars_map and output_vars_map
  std::map<std::string, std::shared_ptr<imperative::VarBase>> current_vars_map;
  for (auto it : inputs) {
    auto& invars = invars_map[it.first];
    invars.reserve(it.second.size());
    for (std::shared_ptr<imperative::VarBase> inp : it.second) {
      PADDLE_ENFORCE_NOT_NULL(inp->var_, "op %s input %s nullptr", op->Type(),
                              inp->Name());

      invars.emplace_back(inp->var_.get());
      if (!stop_gradient) {
        current_vars_map[inp->Name()] = inp;
      }
      VLOG(3) << "input var name: " << inp->Name()
              << " inited: " << inp->var_->IsInitialized()
              << " stop_grad: " << inp->IsStopGradient();
    }
    op->TrackPreOp(it.first, it.second);
  }

  for (const auto& it : *outputs) {
    auto& outvars = outvars_map[it.first];
    const std::vector<std::shared_ptr<imperative::VarBase>>& outputs_tmp =
        it.second;
    outvars.reserve(outputs_tmp.size());
    for (size_t i = 0U; i < outputs_tmp.size(); ++i) {
      // Add weak_ptr to track outputs
      op->outputs_ref.emplace_back(outputs_tmp[i]);
      std::shared_ptr<imperative::VarBase> out = outputs_tmp[i];
      outvars.emplace_back(out->var_.get());
      out->TrackPreOp(op, it.first, i, stop_gradient);
      if (!stop_gradient) {
        current_vars_map[out->Name()] = out;
      }

      VLOG(3) << "output var name: " << out->Name()
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

  // TODO(minqiyang): Support infer var type in imperative mode
  // Run forward op
  VLOG(3) << "tracer running " << op->Type();
  framework::RuntimeContext ctx(invars_map, outvars_map);

  // TODO(panyx0718): Cache p.
  framework::OperatorWithKernel* op_kernel =
      dynamic_cast<framework::OperatorWithKernel*>(op_base.get());
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");

  framework::Scope scope;
  op->place_ = GetExpectedPlace(expected_place, inputs);

  PreparedOp prepared_op = PreparedOp::Prepare(ctx, *op_kernel, op->place_);
  prepared_op.op.RuntimeInferShape(scope, op->place_, ctx);
  prepared_op.func(
      framework::ExecutionContext(prepared_op.op, scope, *prepared_op.dev_ctx,
                                  prepared_op.ctx, prepared_op.kernel_configs));

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
      for (auto it : grad_op_desc->Inputs()) {
        auto& grad_in_vars = op->grad_input_vars_[i][it.first];
        grad_in_vars.reserve(it.second.size());
        for (const std::string& grad_invar : it.second) {
          auto var_it = grad_to_var->find(grad_invar);
          if (var_it == grad_to_var->end()) {
            auto fwd_var_it = current_vars_map.find(grad_invar);
            PADDLE_ENFORCE(fwd_var_it != current_vars_map.end());
            // Forward inputs or outputs.
            grad_in_vars.emplace_back(fwd_var_it->second);
          } else {
            std::shared_ptr<imperative::VarBase> var =
                current_vars_map[var_it->second];
            CreateNoBuffuerGrad(var, prepared_op.GetDeviceContext());
            // Douts.
            var->grads_->SetPreOp(var->PreOp());
            grad_in_vars.emplace_back(var->grads_);
          }
        }
      }

      for (auto it : grad_op_desc->Outputs()) {
        auto& grad_out_vars = op->grad_output_vars_[i][it.first];
        for (const std::string& grad_outvar : it.second) {
          auto var_it = grad_to_var->find(grad_outvar);
          PADDLE_ENFORCE(var_it != grad_to_var->end(),
                         "Could not found the grad op output var, should this "
                         "operator %s's stop gradient be True",
                         op->Type());

          std::shared_ptr<imperative::VarBase> var =
              current_vars_map[var_it->second];
          CreateNoBuffuerGrad(var, prepared_op.GetDeviceContext());
          var->grads_->SetPreOp(var->PreOp());
          grad_out_vars.push_back(var->grads_);
          VLOG(3) << "grads output var name: " << var->name_;
        }
      }
    }
  }
}
}  // namespace imperative
}  // namespace paddle
