// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/new_ir/op_lowering_impl.h"

#include <string>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/hlir/framework/op_lowering_util.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"

#include "paddle/cinn/hlir/framework/new_ir/utils.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/utils/attribute_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/core/ddim.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);

namespace cinn {
namespace hlir {
namespace framework {
namespace newir {

using cinn::hlir::op::ExternalApiRegistry;
using common::Type;
using framework::OpPatternKind;
using framework::StrategyFunction;

namespace details {
ir::Tensor GetTensor(const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto in_shape = phi::vectorize<int>(type_info.dims());
  auto dtype = type_info.dtype();
  std::string input_id = CompatibleInfo::ValueName(value);
  return lang::CreatePlaceHolder(
      in_shape, utils::ConvertIRType(dtype), input_id);
}

std::vector<ir::Tensor> CollectInputTensor(
    const ::pir::Operation* op,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  for (auto in_value : op->operands_source()) {
    VLOG(4) << "input tensor name: " << CompatibleInfo::ValueName(in_value);
    // NOTE(Aurelius84): Need always to create placeholder for input tensor.
    ir::Tensor tensor = details::GetTensor(in_value);
    if (!tensor_map->count(in_value)) {
      // record tensor.
      (*tensor_map)[in_value] = tensor;
      // record func input args
      if (func_args != nullptr) {
        func_args->push_back(tensor);
      }
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

void CollectOutputInfo(::pir::Operation* op,
                       std::vector<Type>* out_types,
                       std::vector<std::vector<int>>* out_shapes) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = CompatibleInfo::ValueName(out_value);
    // group->output_names.push_back(output_id);
    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(utils::ConvertIRType(type_info.dtype()));
    auto out_shape = phi::vectorize<int>(type_info.dims());
    out_shapes->push_back(std::move(out_shape));
  }
}

NodeAttr CollectAttrs(const ::pir::Operation& op) {
  NodeAttr node_attrs;
  VLOG(4) << "op.attributes():" << op.attributes().size();
  auto attrs = utils::ConvertAttributes(op.attributes());
  node_attrs.node_name = CompatibleInfo::OpName(op);
  node_attrs.attr_store = std::move(attrs);

  return node_attrs;
}

}  // namespace details

OpLowererImpl::OpLowererImpl(const Target& target) : target_(target) {}

std::vector<ir::LoweredFunc> OpLowererImpl::Lower(const GroupPtr& group,
                                                  bool apply_op_schedule,
                                                  bool apply_group_schedule,
                                                  bool apply_pass) {
  VLOG(3) << "Lowering Group : " << group->group_id
          << " , Op Pattern : " << group->op_pattern_kind;
  group->input_names.clear();
  group->output_names.clear();
  switch (group->op_pattern_kind) {
    case framework::kElementWise:
    case framework::kBroadcast:
    case framework::kInjective:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::ElementwiseScheduleDetermineFunction);
    case framework::kReduction:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::ReduceScheduleDetermineFunction);
    case framework::kOutFusible:
      LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
    case framework::kNonFusible:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::NonFusibleScheduleDetermineFunction);
    default:
      LOG(FATAL) << "Group Pattern Kind Is Unknown!";
  }
}

bool OpLowererImpl::ElementwiseScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

bool OpLowererImpl::ReduceScheduleDetermineFunction(::pir::Operation* op) {
  // TODO(Aurelius84): Support this.
  // auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  // return op_pattern_dict[op] == framework::kReduction;
  return true;
}

bool OpLowererImpl::NonFusibleScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerGroup(
    const GroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    ScheduleDetermineFunction schedule_determine_func) {
  // 1.Do compute, lower and schedule for each op.
  auto& ops = group->ops;
  if (ops.size() == 1 && ops[0]->name() == "custom_call") {
    return LowerCustomCall(group);
  }
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  bool do_op_schedule = apply_group_schedule || apply_op_schedule;
  std::vector<ir::Expr> func_bodies = LowerOps(ops,
                                               do_op_schedule,
                                               schedule_determine_func,
                                               &group_func_arg_tensors,
                                               &tensor_map);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  // TODO(Aurelius84): Support this.
  // if (apply_group_schedule) {
  //   DoGroupSchedule(ir_sch, group, tensor_map);
  //   VLOG(3) << "After group schedule, ir is: \n"
  //           << ir_sch.GetModule().GetExprs().at(0);
  // }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  return PostProcess(
      group, tensor_map, do_op_schedule, &ir_sch, &group_func_arg_tensors);
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerCustomCall(
    const GroupPtr& group) {
  auto& ops = group->ops;
  CHECK_EQ(ops.size(), 1);
  ::pir::Operation* op = ops[0];
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  std::vector<ir::Tensor> op_func_arg_tensors =
      details::CollectInputTensor(op, nullptr, &tensor_map);
  VLOG(4) << "inputs.size(): " << op_func_arg_tensors.size();

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  details::CollectOutputInfo(op, &out_types, &out_shapes);
  VLOG(4) << "out_types.size(): " << out_types.size();

  NodeAttr node_attrs = details::CollectAttrs(*op);

  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  const hlir::framework::Operator* cinn_op =
      Operator::Get(node_attrs.node_name);
  auto impl = OpStrategy::SelectImpl(cinn_strategy[cinn_op](
      node_attrs, op_func_arg_tensors, out_types, out_shapes, target_));

  // TODO(Arelius84): Support extern API
  std::string external_api;
  // if (node_attrs.attr_store.count("custom_call")) {
  //   external_api =
  //       absl::get<std::string>(node_attrs.attr_store.at("custom_call"));
  // } else {
  //   external_api = ExternalApiRegistry::Global()->GetExternalApi(node,
  //   target_);
  // }
  std::vector<common::CINNValue> compute_args = {
      common::CINNValue(group->fn_name), common::CINNValue(external_api)};
  common::CINNValuePack pack =
      impl->fcompute(common::CINNValuePack{compute_args});
  CHECK_EQ(pack.size(), 1UL);
  // reset input names as extern api input args can't be remove duplicate.
  // group->input_names.clear();
  // for (auto& inode : node->inlinks_in_order()) {
  //   group->input_names.push_back(inode->source()->as<NodeData>()->id());
  // }
  return {pack[0].operator ir::Expr().as_lowered_func_ref()};
}

std::vector<ir::LoweredFunc> OpLowererImpl::PostProcess(
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    bool done_op_schedule,
    ir::IRSchedule* ir_sch,
    std::vector<ir::Tensor>* group_func_arg_tensors) {
  // 1.Prepare function args
  group->input_names.clear();
  std::vector<ir::Argument> group_func_args;
  std::unordered_set<std::string> arg_name_set;
  for (auto& arg_tensor : *group_func_arg_tensors) {
    // input data name.
    group->input_names.push_back(arg_tensor->name);
    // input args
    group_func_args.emplace_back(arg_tensor->buffer, ir::Argument::IO::kInput);
    arg_name_set.insert(arg_tensor->buffer->name);
  }

  group->output_names.clear();
  // FIXME(Aurelius84): Do we need to use output_ops?
  // Currently we regards all ops as output_ops.
  for (auto& op : group->ops) {
    // collect all output tensor.
    for (auto opresult : op->results()) {
      if (tensor_map.count(opresult) == 0) {
        continue;
      }
      auto tensor = tensor_map.at(opresult);
      if (arg_name_set.count(tensor->buffer->name) != 0) {
        continue;
      }
      // output arg tensors
      group_func_arg_tensors->push_back(tensor);
      // output args
      group->output_names.push_back(tensor->name);
      group_func_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      arg_name_set.insert(tensor->buffer->name);
    }
  }
  if (!done_op_schedule) {
    std::unordered_set<std::string> args_set;
    for (auto arg : group_func_args) {
      args_set.insert(arg.name());
    }

    for (auto& tensor_pair : tensor_map) {
      if (args_set.count("_" + tensor_pair.second->name)) {
        continue;
      }
      group_func_arg_tensors->push_back(tensor_pair.second);
      // use the underlying tensor name to be consistent with the argument name
      // in the lowered function
      group->output_names.push_back(tensor_pair.second->name);
      group_func_args.emplace_back(tensor_pair.second->buffer,
                                   ir::Argument::IO::kOutput);
    }
  }

  auto func_body = ir_sch->GetModule().GetExprs().at(0);
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  // 2.Prepare temp buffers
  poly::StageMap stages;
  auto temp_buffers =
      lang::GetTempBuffers(*group_func_arg_tensors, stages, func_body);
  // 3.Building LoweredFunc
  auto func = ir::_LoweredFunc_::Make(group->fn_name,
                                      group_func_args,
                                      ir_sch->GetModule().GetExprs().at(0),
                                      temp_buffers);
  if (!done_op_schedule) {
    func->PrepareBufferCastExprs();
  }
  // 4.Apply low level pass
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
  return {func};
}

std::vector<ir::Expr> OpLowererImpl::LowerOps(
    const std::vector<::pir::Operation*>& ops,
    bool apply_op_schedule,
    ScheduleDetermineFunction schedule_determine_func,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<Expr> func_bodies;
  for (auto* op : ops) {
    // 1.Select Op impl
    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;
    details::CollectOutputInfo(op, &out_types, &out_shapes);
    VLOG(4) << "out_types.size(): " << out_types.size();
    NodeAttr node_attrs = details::CollectAttrs(*op);

    std::vector<ir::Tensor> op_func_arg_tensors =
        details::CollectInputTensor(op, group_func_arg_tensors, tensor_map);
    VLOG(4) << "input size:" << op_func_arg_tensors.size();

    std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    auto op_impl = OpStrategy::SelectImpl(strategy[cinn_op](
        node_attrs, op_func_arg_tensors, out_types, out_shapes, this->target_));

    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs =
        DoOpLower(op_impl, op, tensor_map, &op_func_arg_tensors);

    if (apply_op_schedule && (this->*schedule_determine_func)(op)) {
      // 3.Perform the schedule of Op
      func_bodies.push_back(DoOpSchedule(op_impl, op_func_arg_tensors, funcs));
    } else {
      for (const ir::LoweredFunc& func : funcs) {
        func_bodies.push_back(func->body);
      }
    }
  }

  VLOG(4) << "group_func_arg_tensors.size(): "
          << group_func_arg_tensors->size();

  return func_bodies;
}

std::vector<ir::LoweredFunc> OpLowererImpl::DoOpLower(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    ::pir::Operation* op,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::vector<ir::Tensor>* op_func_arg_tensors) {
  VLOG(4) << "Do lower with Compute, op: " << op->name();
  std::vector<common::CINNValue> cinn_inputs;
  for (const ir::Tensor& tensor : *op_func_arg_tensors) {
    cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
  }
  // set tensor name = operand hash name
  auto op_results = op->results();
  for (const auto& result : op_results) {
    std::string output_id = CompatibleInfo::ValueName(result);
    cinn_inputs.push_back(common::CINNValue(output_id));
  }

  // 1.Do compute
  common::CINNValuePack pack =
      op_impl->fcompute(common::CINNValuePack{cinn_inputs});

  poly::StageMap tmp_stages = pack.back();
  std::string post = "";
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    Expr expr = pack[idx];
    // Insert the output tensor defined by Compute into the tensor_map
    if (pack.size() - 1 > op_results.size()) {
      // Some op may output multiple temp tensors in their Compute
      // definition, but only one output  in the graph, and we use id +
      // "_0"/"_1" as key.
      // FIXME(Aurelius84): It seems that the implementation is relate with
      // string name.
      // (*tensor_map)[op_results[0] + post] = expr.as_tensor_ref();
      // post = "_" + std::to_string(idx);
    } else {
      // If the number of output tensors defined by Compute is less equal than
      // the output node_data on the graph, then there is a one-to-one
      // correspondence, and the redundant output node_data contact empty.
      (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
    }

    // Insert output tensors into function arg
    if (!expr.as_tensor_ref()->buffer.defined() ||
        this->target_ != common::DefaultNVGPUTarget()) {
      op_func_arg_tensors->push_back(expr.as_tensor_ref());
      expr.as_tensor_ref()->WithBuffer();
    }
  }

  VLOG(4) << "op_func_arg_tensors.size(): " << op_func_arg_tensors->size();

  // 2.Do lower
  std::string lower_fn_name = CompatibleInfo::OpFuncName(*op);
  ast_gen_ius::TensorGroup tensor_group =
      ast_gen_ius::ConvertStageMapToTensorGroup(tmp_stages);
  std::vector<ir::LoweredFunc> funcs = lang::LowerToAstVec(
      lower_fn_name, *op_func_arg_tensors, {&tensor_group}, this->target_);
  VLOG(4) << "Lower op: " << lower_fn_name << ", get " << funcs.size()
          << " LoweredFunc:\n";
  if (VLOG_IS_ON(4)) {
    for (auto fun : funcs) {
      VLOG(4) << fun;
    }
  }

  op_func_arg_tensors->clear();
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    CHECK(pack[idx].is_tensor());
    op_func_arg_tensors->push_back(
        pack[idx].operator ir::Expr().as_tensor_ref());
  }

  return funcs;
}

ir::Expr OpLowererImpl::DoOpSchedule(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    const std::vector<ir::Tensor>& op_func_arg_tensors,
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  VLOG(4) << "Do op schedule";
  std::vector<common::CINNValue> schedule_inputs;
  // 1.Collect tensors
  for (const ir::Tensor& op_func_arg_tensor : op_func_arg_tensors) {
    schedule_inputs.push_back(common::CINNValue(op_func_arg_tensor));
  }
  // 2.Collect bodies to be scheduled
  for (const ir::LoweredFunc& func : lowered_funcs) {
    schedule_inputs.push_back(common::CINNValue(func->body));
  }
  // 3.Do schedule on AST
  common::CINNValuePack expr_pack =
      op_impl->fschedule(common::CINNValuePack{schedule_inputs});
  VLOG(4) << "After op schedule: " << expr_pack[0].operator ir::Expr();

  return expr_pack[0].operator ir::Expr();
}

}  // namespace newir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
