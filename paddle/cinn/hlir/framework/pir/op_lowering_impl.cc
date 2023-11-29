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

#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

#include <string>

#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/phi/core/ddim.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);
PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_enable_map_expr_schedule);
PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

using cinn::hlir::op::ExternalApiRegistry;
using common::Type;
using framework::OpPatternKind;
using framework::StrategyFunction;

namespace details {

bool IsInTensorMap(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == CompatibleInfo::ValueName(iter.first)) {
      return true;
    }
  }
  return false;
}

common::Type GetTensorDtype(const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto in_shape = phi::vectorize<int>(type_info.dims());
  auto dtype = type_info.dtype();
  return CompatibleInfo::ConvertIRType(dtype);
}

common::Type GetTensorDtype(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == CompatibleInfo::ValueName(iter.first)) {
      return GetTensorDtype(iter.first);
    }
  }
  VLOG(4) << name << " is not in tensor map, return FP32 by default.";
  return common::F32();
}

ir::Tensor GetTensor(const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto in_shape = phi::vectorize<int>(type_info.dims());
  auto dtype = type_info.dtype();
  std::string input_id = CompatibleInfo::ValueName(value);
  return lang::CreatePlaceHolder(
      in_shape, CompatibleInfo::ConvertIRType(dtype), input_id);
}

std::vector<ir::Tensor> CollectInputTensor(
    const ::pir::Operation* op,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  for (auto in_value : CompatibleInfo::RealOperandSources(*op)) {
    VLOG(4) << "input tensor name: " << CompatibleInfo::ValueName(in_value);
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

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));
    auto out_shape = phi::vectorize<int>(type_info.dims());
    out_shapes->push_back(std::move(out_shape));
  }
}

NodeAttr CollectAttrs(const ::pir::Operation& op) {
  NodeAttr node_attrs;
  VLOG(4) << "op.attributes():" << op.attributes().size();
  auto attrs = CompatibleInfo::ConvertAttributes(op);
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

std::vector<std::pair<ir::SymbolicPredicate, ir::LoweredFunc>>
OpLowererImpl::BucketLower(const GroupPtr& group,
                           bool apply_op_schedule,
                           bool apply_group_schedule,
                           bool apply_pass) {
  // 1.Do compute, lower and schedule for each op.
  auto& ops = group->ops;
  if (ops.size() == 1 && ops[0]->name() == "custom_call") {
    return {{ir::Expr(1), LowerCustomCall(group)[0]}};
  }
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  std::vector<ir::Expr> func_bodies =
      LowerOps(ops,
               apply_op_schedule,
               &OpLowererImpl::DyShapeScheduleDetermineFunction,
               &group_func_arg_tensors,
               &tensor_map,
               &tmp_tensor_info);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  std::vector<std::pair<ir::SymbolicPredicate, ir::Expr>> cond2func_bodies;
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    std::unordered_set<std::string> output_tensor_names;
    std::transform(
        group->output_ops.begin(),
        group->output_ops.end(),
        std::inserter(output_tensor_names, output_tensor_names.begin()),
        [](::pir::Operation* op) {
          return CompatibleInfo::ValueName(op->result(0));
        });
    std::unique_ptr<ir::GroupScheduler> group_scheduler =
        ir::GroupScheduler::Make(
            &ir_sch, output_tensor_names, target_, /* is_dy_shape = */ true);
    group_scheduler->Schedule();
    cond2func_bodies = group_scheduler->GetIRs();
  } else {
    cond2func_bodies.emplace_back(ir::Expr(1),
                                  ir_sch.GetModule().GetExprs()[0]);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<std::pair<ir::Expr, ir::LoweredFunc>> cond2funcs;
  for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
       cond2func_bodies) {
    std::vector<ir::Tensor> group_func_arg_tensors_copy =
        group_func_arg_tensors;
    std::vector<ir::LoweredFunc> funcs =
        PostProcess(group,
                    tensor_map,
                    apply_op_schedule,
                    cond2body.second,
                    &group_func_arg_tensors_copy);
    for (ir::LoweredFunc& func : funcs) {
      cond2funcs.emplace_back(cond2body.first, func);
    }
  }
  return cond2funcs;
}

bool OpLowererImpl::ElementwiseScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

bool OpLowererImpl::ReduceScheduleDetermineFunction(::pir::Operation* op) {
  VLOG(3) << "in ReduceScheduleDetermineFunction";
  return CompatibleInfo::OpKind(*op) == framework::kReduction;
}

bool OpLowererImpl::NonFusibleScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

bool OpLowererImpl::DyShapeScheduleDetermineFunction(::pir::Operation* op) {
  return false;
}

void OpLowererImpl::LowerOpsForMapExpr(
    const GroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
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
    std::vector<ir::LoweredFunc> funcs = DoOpLower(
        op_impl, op, tensor_map, &tmp_tensor_info, &op_func_arg_tensors);

    group->mut_map_expr_ctx()->UpdateOpLoweredFuncKey(op, funcs);
  }
}

/* Most of below codes copies from `PostProcess` function */
std::vector<ir::LoweredFunc> OpLowererImpl::LowerMapExpr(
    const GroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    bool apply_op_schedule,
    bool apply_group_schedule,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  if (FLAGS_cinn_enable_map_expr && FLAGS_cinn_enable_map_expr_schedule) {
    apply_op_schedule = false;
    apply_group_schedule = false;
  }
  VLOG(4) << "FLAGS_cinn_enable_map_expr_schedule = "
          << FLAGS_cinn_enable_map_expr_schedule;
  VLOG(4) << "apply_op_schedule = " << apply_op_schedule;
  VLOG(4) << "apply_group_schedule = " << apply_group_schedule;

  LowerOpsForMapExpr(group, ops, group_func_arg_tensors, tensor_map);

  VLOG(4) << "Begin MapExprToIr";
  ir::Expr func_body = adt::MapExprToIr(group->map_expr_ctx(), target_);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr({func_body});
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    std::unordered_set<std::string> output_tensor_names;
    std::transform(
        group->output_ops.begin(),
        group->output_ops.end(),
        std::inserter(output_tensor_names, output_tensor_names.begin()),
        [](::pir::Operation* node) {
          ::pir::Value node_data = node->result(0);
          return hlir::framework::pir::CompatibleInfo::ValueName(node_data);
        });
    ir::StaticShapeGroupScheduler group_scheduler(
        &ir_sch, output_tensor_names, target_);
    group_scheduler.MapExprSchedule();
    VLOG(3) << "After group schedule, ir is: \n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  return PostProcess(group,
                     *tensor_map,
                     apply_op_schedule,
                     ir_sch.GetModule().GetExprs()[0],
                     group_func_arg_tensors);
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
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  bool do_op_schedule = apply_group_schedule || apply_op_schedule;
  if (FLAGS_cinn_enable_map_expr) {
    return LowerMapExpr(group,
                        ops,
                        /*do_op_schedule=*/do_op_schedule,
                        /*apply_group_schedule=*/apply_group_schedule,
                        &group_func_arg_tensors,
                        &tensor_map);
  }
  std::vector<ir::Expr> func_bodies = LowerOps(ops,
                                               do_op_schedule,
                                               schedule_determine_func,
                                               &group_func_arg_tensors,
                                               &tensor_map,
                                               &tmp_tensor_info);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    DoGroupSchedule(ir_sch, group, tensor_map, tmp_tensor_info);
    VLOG(3) << "After group schedule, ir is: \n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  return PostProcess(group,
                     tensor_map,
                     do_op_schedule,
                     ir_sch.GetModule().GetExprs().at(0),
                     &group_func_arg_tensors);
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
      common::CINNValue(group->FuncName()), common::CINNValue(external_api)};
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
    ir::Expr func_body,
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
  // TODO(phlrain): output values not stable here
  for (auto& op : group->output_ops) {
    // collect all output tensor.
    for (auto opresult : op->results()) {
      if (tensor_map.count(opresult) == 0) {
        continue;
      }
      auto tensor = tensor_map.at(opresult);
      if (arg_name_set.count(tensor->buffer->name) != 0) {
        continue;
      }

      group->output_values.push_back(opresult);
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

  // add fake symbolic args for test
  if (FLAGS_cinn_bucket_compile) {
    group_func_args.emplace_back(ir::_Var_::Make("fake_symbol1", Int(32)),
                                 ir::Argument::IO::kOutput);
    group_func_args.emplace_back(ir::_Var_::Make("fake_symbol2", Int(32)),
                                 ir::Argument::IO::kOutput);
    group->output_names.push_back("fake_symbol1");
    group->output_names.push_back("fake_symbol2");
  }

#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  // 2.Prepare temp buffers
  poly::StageMap stages;
  auto temp_buffers =
      lang::GetTempBuffers(*group_func_arg_tensors, stages, func_body);
  // 3.Building LoweredFunc
  auto func = ir::_LoweredFunc_::Make(
      group->FuncName(), group_func_args, func_body, temp_buffers);
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
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info) {
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
    std::vector<ir::LoweredFunc> funcs = DoOpLower(
        op_impl, op, tensor_map, tmp_tensor_info, &op_func_arg_tensors);

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
    std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info,
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
      if (idx < op_results.size()) {
        (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
      }
      std::string tensor_name = CompatibleInfo::ValueName(op_results[0]) + post;
      VLOG(3) << "Add tmp tensor name for reducer op: " << tensor_name;
      (*tmp_tensor_info)[tensor_name] = expr.as_tensor_ref();
      post = "_" + std::to_string(idx);
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

ir::Expr OpLowererImpl::DoGroupSchedule(
    ir::IRSchedule& ir_sch,
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  // topological order.
  auto ops_set = group->OpSet();
  auto v_consumers = BuildVirtualConsumer(group);
  auto ops_in_order = BFSTopologicalOrderWithPriority(group, v_consumers);
  // find reducer.
  std::unordered_set<::pir::Operation*> ops_inline;
  auto greducer = FindGlobalReducer(ops_in_order);

  // do schedule
  for (auto op : ops_in_order) {
    VLOG(4) << "Try FUSION " << op->name();
    std::string op_name = CompatibleInfo::OpName(*op);
    auto op_kind = CompatibleInfo::OpKind(*op);
    // consumers.
    auto consumers = GetConsumersInSet(op, ops_set);
    auto* reducer = greducer ? FindNearestReducer(op, ops_set) : greducer;
    if (!reducer && greducer) {
      reducer = v_consumers.count(op) ? v_consumers.find(op)->second : reducer;
      if (reducer &&
          CompatibleInfo::OpKind(*reducer) != framework::kReduction) {
        reducer = nullptr;
      }
    }

    auto masters = GetMasters(op, ops_inline, ops_set);
    // TODO(Aurelius84): support inline later.
    if (CanbeInline(op, reducer, consumers, masters, group, ops_set) && false) {
      VLOG(3) << "Before compute inline, ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      auto block = ir_sch.GetBlock(CompatibleInfo::ValueName(op->result(0)));
      ir::ComputeInlineChecker checker(ir_sch, block);
      if (!checker.Check()) {
        checker.BuildDataDependency();
        continue;
      }

      // if exist global reduce node.
      if (greducer) {
        auto loops = ir_sch.GetLoops(CompatibleInfo::ValueName(op->result(0)));
        if (op_kind == framework::kElementWise) {
          ir_sch.FlattenLoops(loops, true);
        } else {
          ir_sch.FlattenLoops(loops, false);
        }
      }

      ir_sch.ComputeInline(block);
      ops_inline.insert(op);
      VLOG(3) << "After compute inline, ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      continue;
    }
    // find master to computeat.
    auto master = GetMasterToComputeAt(
        op, ops_in_order, ops_inline, ops_set, v_consumers);
    std::string op_out_name = CompatibleInfo::ValueName(op->result(0));
    // assign to reducer/master loop.
    if (reducer) {
      VLOG(3) << "Before assign node " << op_name
              << " into vertical link reducer "
              << CompatibleInfo::OpName(*reducer) << ", ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      // if node is vertical with reduce, loop assign reducer.
      LoopAssignReduce(
          ir_sch, op, reducer, this->target_, tensor_map, tmp_tensor_info);
    } else if (greducer) {
      auto greducer_out_shape = CompatibleInfo::ValueShape(greducer->result(0));
      auto op_out_shape = CompatibleInfo::ValueShape(op->result(0));
      if (CompatibleInfo::ShapeProduct(greducer_out_shape) !=
          CompatibleInfo::ShapeProduct(op_out_shape)) {
        LoopAssignReduce(
            ir_sch, op, greducer, this->target_, tensor_map, tmp_tensor_info);
      }
    } else if (master) {
      VLOG(3) << "Before assign node " << op_name
              << " into horizontal link reducer, ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      // if node is horizontal with reduce or node is reduce, loop assign
      // master.
      auto loops = ir_sch.GetLoops(op_out_name);
      ir_sch.Fuse(loops);

      if (master && op_kind != framework::kReduction) {
        auto master_loops =
            ir_sch.GetLoops(CompatibleInfo::ValueName(master->result(0)));
        std::vector<int> splits;
        for (auto loop : master_loops) {
          splits.push_back(loop.As<ir::For>()->extent.as_int32());
        }
        loops = ir_sch.GetLoops(op_out_name);
        ir_sch.Split(loops[0], splits);
      }
    }
    VLOG(3) << "Before loop fusion, ir is:\n"
            << ir_sch.GetModule().GetExprs().at(0);
    // do loop fuse.
    LoopComputeAt(ir_sch,
                  op,
                  master ? master : ops_in_order.front(),
                  group,
                  tensor_map,
                  tmp_tensor_info);
    VLOG(3) << "After loop fusion, ir is:\n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // do vectorize
  auto all_blocks = ir_sch.GetAllBlocks();
  VLOG(4) << "Size of blocks: " << all_blocks.size();
  VLOG(4) << "Op Pattern : " << group->op_pattern_kind;

  // only support first block?
  auto block = all_blocks[0];

  if (block->as<ir::ScheduleBlockRealize>() == nullptr ||
      block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>() == nullptr) {
    std::string err_msg =
        "Group scheduling, the Expr is not wrapped by ScheduleBlockRealize or "
        "ScheduleBlock, cannot be scheduled.";
    std::ostringstream detail_info;
    detail_info << "Expr:\n";
    detail_info << block;
    throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                              err_msg,
                              detail_info.str(),
                              __FILE__,
                              __LINE__);
  }
  auto is_tensor_block = true;
  auto tensor_name = block->as<ir::ScheduleBlockRealize>()
                         ->schedule_block->as<ir::ScheduleBlock>()
                         ->name;
  if (!details::IsInTensorMap(tensor_name, tensor_map)) {
    is_tensor_block = false;
  }
  if (FLAGS_cinn_use_cuda_vectorize && is_tensor_block &&
      (group->op_pattern_kind == framework::kElementWise ||
       group->op_pattern_kind == framework::kInjective ||
       group->op_pattern_kind == framework::kBroadcast)) {
    // auto loops = ir_sch.GetLoops(GetNodeData(node)->id());
    auto loops = ir_sch.GetLoops(block);
    VLOG(4) << "Op Pattern : " << loops.size();
    if (loops.size() >= 1) {
      VLOG(4) << "Before vectorize, ir is: \n"
              << ir_sch.GetModule().GetExprs().at(0);
      auto loop_inner = loops.back();
      int vector_width = 1;
      auto psize = ir::GetLoopExtent(loop_inner);
      auto dtype = details::GetTensorDtype(tensor_name, tensor_map);
      VLOG(4) << tensor_name << " dtype " << dtype;
      if (psize % 8 == 0 && (dtype.is_float16() || dtype.is_bfloat16())) {
        vector_width = 8;
      } else if (psize % 4 == 0) {
        vector_width = 4;
      } else if (psize % 2 == 0) {
        vector_width = 2;
      }
      if (vector_width > 1) {
        auto splited = ir_sch.Split(loop_inner, {-1, vector_width});
        splited[0].As<ir::For>()->set_bind_info(
            loop_inner.As<ir::For>()->bind_info());
        splited[1].As<ir::For>()->set_serial();
        ir_sch.Vectorize(splited[1], vector_width);
      }
      VLOG(4) << "After vectorize, ir is: \n"
              << ir_sch.GetModule().GetExprs().at(0);
    }
  }

  VLOG(3) << "Before Sync IRLowerOp schedule, ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
  SyncThreadWithShared(ir_sch, group, ops_inline, ops_set, tensor_map);
  VLOG(4) << "After IRSchedule,  ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
  return ir_sch.GetModule().GetExprs().at(0);
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
