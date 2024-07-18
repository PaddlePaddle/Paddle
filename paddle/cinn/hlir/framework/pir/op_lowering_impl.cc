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
#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/check_tensor_buffer_map.h"
#include "paddle/cinn/optim/eliminate_common_global_memory_read.h"
#include "paddle/cinn/optim/if_fusion.h"
#include "paddle/cinn/optim/rearrange_load_instruction.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);
PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_enable_map_expr_schedule);
PD_DECLARE_bool(cinn_bucket_compile);
PD_DECLARE_bool(cinn_new_group_scheduler);
PD_DECLARE_bool(cinn_check_tensor_buffer_map);
const int default_priority = 100;

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

using cinn::common::Type;
using framework::OpPatternKind;
using framework::StrategyFunction;

namespace details {

NodeAttr CollectAttrs(const ::pir::Operation& op) {
  NodeAttr node_attrs;
  VLOG(4) << "op.attributes():" << op.attributes().size();
  auto attrs = CompatibleInfo::ConvertAttributes(op);
  node_attrs.node_name = CompatibleInfo::OpName(op);
  node_attrs.attr_store = std::move(attrs);

  return node_attrs;
}

std::optional<std::vector<ir::Expr>> GetTensorValueFromShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  if (!shape_or_data.data()) return std::nullopt;
  std::vector<ir::Expr> result;
  result.reserve(shape_or_data.data()->size());
  for (const auto& data : *shape_or_data.data()) {
    result.push_back(common::DimExprConverter().ConvertToIrExpr(data));
  }
  return result;
}

}  // namespace details

std::shared_ptr<GroupInfo> OpLowererImpl::GetGroupInfo(
    const FusionGroupInfo& fusion_group_info,
    const OpLoweringGroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  std::shared_ptr<GroupInfo> group_info = std::make_shared<GroupInfo>();
  group_info->data_space = fusion_group_info.loop_ranges;
  group_info->reduce_axis = fusion_group_info.reduce_axis;
  group_info->reduce_var_names =
      std::set<std::string>(fusion_group_info.reduce_var_name.begin(),
                            fusion_group_info.reduce_var_name.end());

  for (auto& val : group->output_values()) {
    group_info->direct_output_var_names.insert(ValueName(val));
  }

  group->WalkOps([&group_info](::pir::Operation* op) {
    if (CompatibleInfo::OpKind(*op) == OpPatternKind::kReduction) {
      group_info->raw_reduce_axis = cinn::fusion::GetReduceAxisIdx(op);
      group_info->raw_data_rank =
          cinn::fusion::GetCompitableRank(op->operand_source(0));
    }
  });
  return group_info;
}

std::shared_ptr<GroupInfo> OpLowererImpl::GetGroupInfo(
    const OpLoweringGroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  std::shared_ptr<GroupInfo> group_info = std::make_shared<GroupInfo>();
  group_info->data_space = group->loop_ranges();
  group_info->reduce_axis = group->reduce_axis();
  for (auto op : group->ops()) {
    if (CompatibleInfo::OpKind(*op) == OpPatternKind::kReduction) {
      group_info->reduce_var_names.insert(ValueName(op->result(0)));
    }
  }

  for (auto& op : group->output_ops()) {
    group_info->direct_output_var_names.insert(ValueName(op->result(0)));
    // collect all output tensor.
    for (auto opresult : op->results()) {
      if (tensor_map.count(opresult) == 0) {
        continue;
      }
      group_info->direct_output_var_names.insert(ValueName(opresult));
    }
  }

  for (const auto& val : group->output_values()) {
    group_info->direct_output_var_names.insert(ValueName(val));
  }
  return group_info;
}

OpLowererImpl::OpLowererImpl(const Target& target) : target_(target) {
  name_gene_ = new PrettyNamer();
}

std::vector<ir::LoweredFunc> OpLowererImpl::Lower(
    const OpLoweringGroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    bool apply_pass) {
  VLOG(3) << "Lowering Group : " << group->group_id()
          << " , Op Pattern : " << group->op_pattern_kind();
  group->mut_input_names().clear();
  group->mut_output_names().clear();
  switch (group->op_pattern_kind()) {
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
      PADDLE_THROW(phi::errors::Unimplemented(
          "Group Pattern Kind kOutFusible Is Not Implemented!"));
    case framework::kNonFusible:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::NonFusibleScheduleDetermineFunction);
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("Group Pattern Kind Is Unknown!"));
  }
}
BucketLoweredFuncsWrapper OpLowererImpl::BucketLower(
    const OpLoweringGroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    bool apply_pass) {
  VLOG(4) << "BucketLower Group : \n" << *group;
  // 1.Do compute, lower and schedule for each op.
  const auto& ops = group->ops();
  if (ops.size() == 1 && ops[0]->name() == "custom_call") {
    return {{std::make_tuple(
                ir::Expr(1), LowerCustomCall(group)[0], default_priority)},
            ir::LoweredFunc()};
  }
  auto X86Expr = LowerX86(group, ops, apply_op_schedule);
  VLOG(3) << "After x86 lower, ir is: \n" << X86Expr;

  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  std::vector<ir::Expr> func_bodies =
      LowerOps(group,
               ops,
               apply_op_schedule,
               &OpLowererImpl::DyShapeScheduleDetermineFunction,
               &group_func_arg_tensors,
               &tensor_map,
               &tmp_tensor_info);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(func_bodies, "BucketLower LowerOps");
    VLOG(3) << "LowerOps tensor-buffer map check succeed";
  }

  // =========== OpFusion ============

  // VLOG(4) << "Bucket Lower output values is : " << group->output_values();
  func_bodies = OperationFusion(ops, func_bodies, group->output_values());
  const auto& fusion_group_info = GetFusionGroupInfo(func_bodies);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(func_bodies, "BucketLower OpFusion");
    VLOG(3) << "OpFusion tensor-buffer map check succeed";
  }

  // =========== CodeGen And Optimizer ================

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir_sch.MergeExprs();
  std::vector<std::pair<ir::SymbolicPredicate, ir::Expr>> cond2func_bodies;
  std::vector<int> priorities;
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    optim::CheckTensorBufferMap(ir_sch.GetModule().GetExprs(),
                                "BucketLower MergeExprs");
    VLOG(3) << "MergeExprs tensor-buffer map check succeed";
  }

  std::unordered_set<::pir::Value> inner_genevalue;
  std::unordered_set<::pir::Operation*> ops_set(ops.begin(), ops.end());
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      inner_genevalue.insert(op->result(i));
    }
  }

  if (apply_group_schedule) {
    std::unordered_set<std::string> output_tensor_names;
    for (auto value : group->GetGroupOutputValues()) {
      output_tensor_names.insert(ValueName(value));
    }

    std::shared_ptr<GroupInfo> group_info =
        GetGroupInfo(fusion_group_info, group, tensor_map);
    std::unique_ptr<ir::GroupScheduler> group_scheduler =
        ir::GroupScheduler::Make(&ir_sch,
                                 output_tensor_names,
                                 target_,
                                 /* is_dy_shape = */ true,
                                 group_info);

    VLOG(4) << "Start apply group_scheduler->Schedule()";
    group_scheduler->Schedule();
    VLOG(4) << "End   apply group_scheduler->Schedule()";

    cond2func_bodies = group_scheduler->GetIRs();
    VLOG(4) << "End   group_scheduler->GetIRs";

    priorities = group_scheduler->GetPriorities();
    VLOG(4) << "End group_scheduler->GetPriorities";

  } else {
    cond2func_bodies.emplace_back(ir::Expr(true),
                                  ir_sch.GetModule().GetExprs()[0]);
    priorities.emplace_back(default_priority);
  }

  // The last func is stored as a kernel on x86
  cond2func_bodies.emplace_back(ir::Expr(true), X86Expr);

  if (FLAGS_cinn_check_tensor_buffer_map) {
    for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
         cond2func_bodies) {
      optim::CheckTensorBufferMap(cond2body.second, "BucketLower schedule");
    }
    VLOG(3) << "Schedule tensor-buffer map check succeed";
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Expr> scheduled_func_bodies;
  for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
       cond2func_bodies) {
    scheduled_func_bodies.push_back(cond2body.second);
  }
  std::vector<ir::Tensor> group_func_arg_tensors_copy = group_func_arg_tensors;
  std::vector<ir::Argument> group_func_args;
  std::vector<ir::Tensor> infer_shape_tensor_args;
  std::vector<ir::LoweredFunc> funcs = PostProcess(group,
                                                   tensor_map,
                                                   apply_group_schedule,
                                                   {scheduled_func_bodies},
                                                   &group_func_arg_tensors_copy,
                                                   &group_func_args,
                                                   &infer_shape_tensor_args);
  if (FLAGS_cinn_check_tensor_buffer_map) {
    for (ir::LoweredFunc& func : funcs) {
      optim::CheckTensorBufferMap(Expr(func), "BucketLower PostProcess");
    }
    VLOG(3) << "PostProcess tensor-buffer map check succeed";
  }
  PADDLE_ENFORCE_EQ(funcs.size(),
                    cond2func_bodies.size(),
                    phi::errors::InvalidArgument(
                        "The size of funcs and cond2func_bodies should be "
                        "the same."));
  PADDLE_ENFORCE_EQ(
      funcs.size(),
      priorities.size() + 1,
      phi::errors::InvalidArgument("The size of funcs should equals to the "
                                   "size of priorities plus one."));
  BucketLoweredFuncsWrapper funcs_wrapper;
  for (int i = 0; i < funcs.size() - 1; ++i) {
    funcs_wrapper.predicate2funcs.emplace_back(
        std::make_tuple(cond2func_bodies[i].first, funcs[i], priorities[i]));
  }
  // The last func is x86 kernel.
  for (size_t i = funcs.size() - 1; i < funcs.size(); ++i) {
    funcs[i]->name = funcs[i]->name + "_CX86";
    funcs_wrapper.predicate2funcsCX86.emplace_back(cond2func_bodies[i].first,
                                                   funcs[i]);
  }
  funcs_wrapper.infer_shape_func =
      GenerateInferShapeFunc(group, infer_shape_tensor_args, group_func_args);

  VLOG(4) << "End This function.";
  return funcs_wrapper;
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
    const OpLoweringGroupPtr& group,
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

    CollectOutputInfo(op, &out_types, &out_shapes, group);
    VLOG(4) << "out_types.size(): " << out_types.size();
    NodeAttr node_attrs = details::CollectAttrs(*op);

    std::vector<ir::Tensor> op_func_arg_tensors =
        CollectInputTensor(group, op, group_func_arg_tensors, tensor_map);
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
    const OpLoweringGroupPtr& group,
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

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Argument> group_func_args;
  std::vector<ir::Tensor> infer_shape_tensor_args;
  return PostProcess(group,
                     *tensor_map,
                     apply_op_schedule,
                     {ir_sch.GetModule().GetExprs()[0]},
                     group_func_arg_tensors,
                     &group_func_args,
                     &infer_shape_tensor_args);
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerGroup(
    const OpLoweringGroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    ScheduleDetermineFunction schedule_determine_func) {
  // 1.Do compute, lower and schedule for each op.
  const auto& ops = group->ops();
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
  std::vector<ir::Expr> func_bodies =
      LowerOps(group,
               ops,
               do_op_schedule,
               &OpLowererImpl::DyShapeScheduleDetermineFunction,
               &group_func_arg_tensors,
               &tensor_map,
               &tmp_tensor_info);

  // func_bodies = TrivialOpFusion(ops, func_bodies);
  std::unordered_set<::pir::Value> inner_genevalue;
  std::unordered_set<::pir::Operation*> ops_set(ops.begin(), ops.end());
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      inner_genevalue.insert(op->result(i));
    }
  }

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  std::shared_ptr<ir::IRSchedule> ir_sch =
      std::make_shared<ir::IRSchedule>(mod_expr);

  auto have_dy_shape = false;
  for (auto d : group->loop_ranges()) {
    if (d < 0) {
      have_dy_shape = true;
    }
  }
  if (have_dy_shape) {
    ir_sch = std::make_shared<ir::IRSchedule>(
        mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  }
  ir_sch->MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch->GetModule().GetExprs().at(0);
  // if (apply_group_schedule) {
  DoGroupSchedule(*(ir_sch.get()), group, tensor_map, tmp_tensor_info);
  VLOG(3) << "After group schedule, ir is: \n"
          << ir_sch->GetModule().GetExprs().at(0);
  // }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Argument> group_func_args;
  std::vector<ir::Tensor> infer_shape_args;
  return PostProcess(group,
                     tensor_map,
                     do_op_schedule,
                     {ir_sch->GetModule().GetExprs().at(0)},
                     &group_func_arg_tensors,
                     &group_func_args,
                     &infer_shape_args);
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerCustomCall(
    const OpLoweringGroupPtr& group) {
  const auto& ops = group->ops();
  PADDLE_ENFORCE_EQ(
      ops.size(),
      1,
      phi::errors::InvalidArgument("Custom call group should have only "
                                   "one op"));
  ::pir::Operation* op = ops[0];
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  std::vector<ir::Tensor> op_func_arg_tensors =
      CollectInputTensor(group, op, nullptr, &tensor_map);
  VLOG(4) << "inputs.size(): " << op_func_arg_tensors.size();

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  CollectOutputInfo(op, &out_types, &out_shapes, group);
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
  std::vector<cinn::common::CINNValue> compute_args = {
      cinn::common::CINNValue(group->FuncName()),
      cinn::common::CINNValue(external_api)};
  cinn::common::CINNValuePack pack =
      impl->fcompute(cinn::common::CINNValuePack{compute_args});
  PADDLE_ENFORCE_EQ(
      pack.size(),
      1UL,
      phi::errors::InvalidArgument("The size of pack should be 1."));
  // reset input names as extern api input args can't be remove duplicate.
  // group->input_names.clear();
  // for (auto& inode : node->inlinks_in_order()) {
  //   group->input_names.push_back(inode->source()->as<NodeData>()->id());
  // }
  return {pack[0].operator ir::Expr().as_lowered_func_ref()};
}

std::unordered_set<std::string> CollectStoreBufferNames(
    const std::vector<ir::Expr>& func_bodies) {
  std::unordered_set<std::string> buffer_names;
  std::vector<ir::Expr> blocks = ir::analyzer::GetAllBlocks(func_bodies);
  for (const ir::Expr& block : blocks) {
    ir::Tensor tensor = ir::analyzer::GetStoreTensorOfSBlock(block);
    if (tensor->buffer.defined()) {
      buffer_names.insert(tensor->buffer->name);
    }
  }
  return buffer_names;
}

std::vector<ir::LoweredFunc> OpLowererImpl::PostProcess(
    const OpLoweringGroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    bool done_op_schedule,
    std::vector<ir::Expr> func_bodies,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::vector<ir::Argument>* group_func_args,
    std::vector<ir::Tensor>* infer_shape_arg_tensor) {
  // 1.Prepare function args
  group->mut_input_names().clear();
  std::unordered_set<std::string> store_buffer_names =
      CollectStoreBufferNames(func_bodies);
  std::unordered_set<std::string> arg_name_set;
  for (auto& arg_tensor : *group_func_arg_tensors) {
    // input data name.
    group->mut_input_names().push_back(arg_tensor->name);
    // args
    ir::Argument::IO io_type =
        store_buffer_names.count(arg_tensor->buffer->name) > 0
            ? ir::Argument::IO::kOutput
            : ir::Argument::IO::kInput;
    (*group_func_args).emplace_back(arg_tensor->buffer, io_type);
    arg_name_set.insert(arg_tensor->buffer->name);
  }

  group->mut_output_names().clear();

  // collect all output tensor.
  for (auto op_result : group->output_values()) {
    if (tensor_map.count(op_result) == 0) {
      continue;
    }
    auto tensor = tensor_map.at(op_result);
    if (group->HasShapeOrDataExprs(op_result)) {
      tensor->shape.clear();
      for (size_t i = 0;
           i < group->GetShapeOrDataExprs(op_result).shape().size();
           ++i) {
        ir::Dim t(tensor->name,
                  group->GetShapeOrDataExprs(op_result).shape()[i]);
        tensor->shape.push_back(t->dim_expr);
      }
    }
    infer_shape_arg_tensor->push_back(tensor);

    if (arg_name_set.count(tensor->buffer->name) != 0) {
      continue;
    }

    // output arg tensors
    group_func_arg_tensors->push_back(tensor);
    // output args
    group->mut_output_names().push_back(tensor->name);
    (*group_func_args).emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
    arg_name_set.insert(tensor->buffer->name);
  }

  if (!done_op_schedule) {
    std::unordered_set<std::string> args_set;
    for (auto arg : (*group_func_args)) {
      args_set.insert(arg.name());
    }
    for (const auto& op : group->ops()) {
      // collect all output tensor.
      for (auto opresult : op->results()) {
        if (tensor_map.count(opresult) == 0) {
          continue;
        }
        auto tensor = tensor_map.at(opresult);
        if (args_set.count("_" + tensor->name) != 0) {
          continue;
        }
        group->mut_output_values().push_back(opresult);
        group_func_arg_tensors->push_back(tensor);
        group->mut_output_names().push_back(tensor->name);
        group_func_args->emplace_back(tensor->buffer,
                                      ir::Argument::IO::kOutput);
      }
    }
  }

  std::map<int, CINNKernelInfo::ArgDimIdx> mps;
  // update args for dynamic dim
  int num_tensor_args = static_cast<int>(group_func_args->size());
  int non_tensor_arg_idx = group_func_args->size();
  std::unordered_set<std::string> int_args_set;
  for (int tensor_arg_idx = 0; tensor_arg_idx < num_tensor_args;
       tensor_arg_idx++) {
    auto tensor_dim = (*group_func_arg_tensors)[tensor_arg_idx]->sym_shape;
    int tensor_dim_size = tensor_dim.size();
    for (int tensor_arg_dim_idx = 0; tensor_arg_dim_idx < tensor_dim_size;
         tensor_arg_dim_idx++) {
      if (tensor_dim[tensor_arg_dim_idx]->IsUniSymbolic()) {
        const std::string symbol_name =
            tensor_dim[tensor_arg_dim_idx]->ToString();
        if (int_args_set.count(symbol_name) != 0) {
          continue;
        }
        int_args_set.insert(symbol_name);
        group_func_args->emplace_back(
            ir::_Var_::Make(symbol_name, cinn::common::Int(64)));
        group->mut_int_args_map()[non_tensor_arg_idx++] = {tensor_arg_idx,
                                                           tensor_arg_dim_idx};
        VLOG(4) << "device kernel func's " << symbol_name << " is from "
                << tensor_arg_idx << ".shape(" << tensor_arg_dim_idx << ")";
      }
    }
  }
  std::vector<ir::LoweredFunc> lowered_funcs;
  for (int i = 0; i < func_bodies.size(); ++i) {
    ir::Expr func_body = func_bodies[i];
    optim::EliminateDeadScheduleBlock(&(func_body), group->output_names());
    if (i != func_bodies.size() - 1) {
      cinn::common::DefaultDeviceTarget().arch.Match(
          [&](std::variant<common::UnknownArch,
                           common::X86Arch,
                           common::ARMArch>) {},
          [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
            optim::EliminateCommonGlobalMemoryRead(&(func_body));
            optim::OptimizeExprGPU(&(func_body));
#endif
          },
          [&](common::HygonDCUArchHIP) {
#ifdef CINN_WITH_HIP
            optim::EliminateCommonGlobalMemoryRead(&(func_body));
            optim::OptimizeExprGPU(&(func_body));
#endif
          });
    }

    // 2.Prepare temp buffers
    auto temp_buffers =
        lang::GetTempBuffers(*group_func_arg_tensors, func_body);
    // 3.Building LoweredFunc
    auto func = ir::_LoweredFunc_::Make(
        group->FuncName(), *group_func_args, func_body, temp_buffers);
    if (!done_op_schedule) {
      func->PrepareBufferCastExprs();
    }
    // 4.Apply low level pass
    if (i != func_bodies.size() - 1) {
      func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
      optim::RearrangeLoadInstruction(&(func->body));
    } else {
      func = optim::Optimize(Expr(func), common::DefaultHostTarget(), false)
                 .as_lowered_func_ref();
    }
    lowered_funcs.push_back(std::move(func));
  }

  return lowered_funcs;
}

std::vector<ir::Expr> OpLowererImpl::LowerOps(
    const OpLoweringGroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    bool apply_op_schedule,
    ScheduleDetermineFunction schedule_determine_func,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<Expr> func_bodies;
  std::unordered_set<::pir::Value> inner_used_value;
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      inner_used_value.insert(op->operand_source(i));
    }
  }

  std::unordered_set<::pir::Operation*> not_used_op;
  for (auto* op : ops) {
    bool used = false;
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (inner_used_value.count(op->result(i))) {
        used = true;
        break;
      }
    }

    if (!used) {
      not_used_op.insert(op);
    }
  }

  for (auto* op : ops) {
    VLOG(4) << "start lowering op:" << op->name() << " id: " << op->id();
    std::string cinn_op_name = CompatibleInfo::OpName(*op);

    VLOG(4) << "cinn op name " << cinn_op_name << std::endl;

    // 1.Select Op impl
    std::vector<ir::Tensor> op_func_arg_tensors =
        CollectInputTensor(group, op, group_func_arg_tensors, tensor_map);
    VLOG(4) << "input size:" << op_func_arg_tensors.size();

    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    std::shared_ptr<OpImpl> op_impl = nullptr;
    if (FLAGS_cinn_bucket_compile) {
      std::vector<Type> out_types;
      std::vector<std::vector<ir::Dim>> out_shapes;
      CollectOutputInfo(op, &out_types, &out_shapes, group);

      PADDLE_ENFORCE_EQ(out_types.size(),
                        out_shapes.size(),
                        phi::errors::InvalidArgument(
                            "The size of out_types and out_shapes should be "
                            "the same."));
      VLOG(4) << "out_types.size(): " << out_types.size();
      NodeAttr node_attrs = details::CollectAttrs(*op);
      auto& strategy_map =
          Operator::GetAttrs<StrategyFunctionSymbolic>("CINNStrategySymbolic");
      StrategyFunctionSymbolic strategy = strategy_map[cinn_op];
      CHECK(static_cast<bool>(strategy))
          << " cinn_op_name: " << cinn_op_name
          << " has no CINNStrategySymbolic registered.";
      op_impl = OpStrategy::SelectImpl(strategy(node_attrs,
                                                op_func_arg_tensors,
                                                out_types,
                                                out_shapes,
                                                this->target_));
    } else {
      std::vector<Type> out_types;
      std::vector<std::vector<int>> out_shapes;
      CollectOutputInfo(op, &out_types, &out_shapes, group);
      VLOG(4) << "out_types.size(): " << out_types.size();
      NodeAttr node_attrs = details::CollectAttrs(*op);
      op_impl = OpStrategy::SelectImpl(strategy[cinn_op](node_attrs,
                                                         op_func_arg_tensors,
                                                         out_types,
                                                         out_shapes,
                                                         this->target_));
    }
    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs = DoOpLower(
        op_impl, op, tensor_map, tmp_tensor_info, &op_func_arg_tensors);

    for (const ir::LoweredFunc& func : funcs) {
      func_bodies.push_back(func->body);
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
  std::vector<cinn::common::CINNValue> cinn_inputs;
  for (const ir::Tensor& tensor : *op_func_arg_tensors) {
    cinn_inputs.push_back(cinn::common::CINNValue(ir::Expr(tensor)));
  }

  // set tensor name = operand hash name
  auto op_results = op->results();
  for (const auto& result : op_results) {
    std::string output_id = ValueName(result);
    cinn_inputs.push_back(cinn::common::CINNValue(output_id));
  }

  // 1.Do compute
  cinn::common::CINNValuePack pack =
      op_impl->fcompute(cinn::common::CINNValuePack{cinn_inputs});

  std::string post = "";
  std::vector<ir::Tensor> stage_tensors;
  for (int idx = 0; idx < pack.size(); ++idx) {
    Expr expr = pack[idx];
    // Insert the output tensor defined by Compute into the tensor_map
    if (pack.size() - 1 > op_results.size()) {
      // Some op may output multiple temp tensors in their Compute
      // definition, but only one output  in the graph, and we use id +
      // "_0"/"_1" as key.
      if (idx < op_results.size()) {
        (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
      }
      std::string tensor_name = ValueName(op_results[0]) + post;
      VLOG(3) << "Add tmp tensor name for reducer op: " << tensor_name;
      (*tmp_tensor_info)[tensor_name] = expr.as_tensor_ref();
      post = "_" + std::to_string(idx);
    } else {
      // If the number of output tensors defined by Compute is less equal than
      // the output node_data on the graph, then there is a one-to-one
      // correspondence, and the redundant output node_data contact empty.
      (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();

      stage_tensors.push_back(expr.as_tensor_ref());
    }

    // Insert output tensors into function arg
    target_.arch.Match(
        [&](common::NVGPUArch) {
          if (!expr.as_tensor_ref()->buffer.defined()) {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
            expr.as_tensor_ref()->WithBuffer();
          } else {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
          }
        },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) {
          op_func_arg_tensors->push_back(expr.as_tensor_ref());
          expr.as_tensor_ref()->WithBuffer();
        },
        [&](common::HygonDCUArchHIP) {
          if (!expr.as_tensor_ref()->buffer.defined()) {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
            expr.as_tensor_ref()->WithBuffer();
          } else {
            op_func_arg_tensors->push_back(expr.as_tensor_ref());
          }
        });
  }

  VLOG(4) << "op_func_arg_tensors.size(): " << op_func_arg_tensors->size();

  // 2.Do lower
  std::string lower_fn_name = CompatibleInfo::OpFuncName(*op);

  // using output value build tensor group
  ast_gen_ius::TensorGroup tensor_group(stage_tensors);
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
  std::vector<cinn::common::CINNValue> schedule_inputs;
  // 1.Collect tensors
  for (const ir::Tensor& op_func_arg_tensor : op_func_arg_tensors) {
    schedule_inputs.push_back(cinn::common::CINNValue(op_func_arg_tensor));
  }
  // 2.Collect bodies to be scheduled
  for (const ir::LoweredFunc& func : lowered_funcs) {
    schedule_inputs.push_back(cinn::common::CINNValue(func->body));
  }
  // 3.Do schedule on AST
  cinn::common::CINNValuePack expr_pack =
      op_impl->fschedule(cinn::common::CINNValuePack{schedule_inputs});
  VLOG(4) << "After op schedule: " << expr_pack[0].operator ir::Expr();

  return expr_pack[0].operator ir::Expr();
}

ir::Expr OpLowererImpl::DoGroupSchedule(
    ir::IRSchedule& ir_sch,
    const OpLoweringGroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  VLOG(3) << "using StaticShapeGroupScheduler to schedule group.";
  bool have_dy_shape = false;
  for (auto d : group->loop_ranges()) {
    if (d < 0) {
      have_dy_shape = true;
    }
  }

  std::shared_ptr<GroupInfo> group_info = GetGroupInfo(group, tensor_map);

  std::unordered_set<std::string> output_tensor_names;
  for (auto value : group->GetGroupOutputValues()) {
    output_tensor_names.insert(ValueName(value));
  }
  std::unique_ptr<ir::GroupScheduler> group_scheduler =
      ir::GroupScheduler::Make(&ir_sch,
                               output_tensor_names,
                               target_,
                               /* is_dy_shape = */ true,
                               group_info);
  group_scheduler->Schedule();
  return ir_sch.GetModule().GetExprs().at(0);
}

ir::Tensor OpLowererImpl::GetTensor(const OpLoweringGroupPtr& group,
                                    const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto dtype = type_info.dtype();
  std::string input_id = ValueName(value);

  auto ForEachDimExpr = [&](const auto& DoEach) {
    const auto& dims = type_info.dims();
    if (::common::contain_unknown_dim(dims)) {  // dynamic shape
      const auto& sym_vec = group->GetShapeOrDataExprs(value).shape();
      for (const auto& dim_expr : sym_vec) {
        DoEach(dim_expr);
      }
    } else {  // static shape
      for (int i = 0; i < dims.size(); ++i) {
        DoEach(::symbol::DimExpr{dims[i]});
      }
    }
  };

  if (FLAGS_cinn_bucket_compile) {
    std::vector<ir::Dim> sym_shape;
    ForEachDimExpr(
        [&](const auto& sym) { sym_shape.emplace_back(input_id, sym); });
    if (sym_shape.empty()) {
      sym_shape.emplace_back(input_id, symbol::DimExpr{1});
    }
    auto tensor = lang::CreatePlaceHolder(
        sym_shape, CompatibleInfo::ConvertIRType(dtype), input_id);
    auto IsIntType = [](const ::pir::Type& t) {
      return t.isa<::pir::Int32Type>() || t.isa<::pir::Int64Type>();
    };
    if (IsIntType(dtype) && group->HasShapeOrDataExprs(value)) {
      const auto& tensor_value = details::GetTensorValueFromShapeOrData(
          group->GetShapeOrDataExprs(value));
      if (tensor_value.has_value()) {
        tensor->set_value(*tensor_value);
      }
    }
    return tensor;
  } else {
    auto shape = ::common::vectorize<int>(type_info.dims());
    return lang::CreatePlaceHolder(
        shape, CompatibleInfo::ConvertIRType(dtype), input_id);
  }
}

std::vector<ir::Tensor> OpLowererImpl::CollectInputTensor(
    const OpLoweringGroupPtr& group,
    const ::pir::Operation* op,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  for (auto in_value : CompatibleInfo::RealOperandSources(*op)) {
    VLOG(4) << "input tensor name: " << ValueName(in_value);
    ir::Tensor tensor = GetTensor(group, in_value);
    VLOG(4) << "shape: " << tensor->shape;
    VLOG(4) << "sym_shape: " << tensor->sym_shape;

    if (!tensor_map->count(in_value)) {
      // record tensor.
      (*tensor_map)[in_value] = tensor;
      // record func input args
      if (func_args != nullptr) {
        func_args->push_back(tensor);
      }
    } else {
      // TODO(6clc): After supporting symbolic calculation,
      // 1. Check that the shape of the tensor with the same name is the same
      // size
      // 2. Or make the symbol expression in compute output tensor consistent
      //    with the one inferred in shape_analysis
      (*tensor_map)[in_value]->sym_shape = tensor->sym_shape;
      (*tensor_map)[in_value]->shape = tensor->shape;
      (*tensor_map)[in_value]->sym_domain = tensor->sym_domain;
      (*tensor_map)[in_value]->domain = tensor->domain;
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

void OpLowererImpl::CollectOutputInfo(::pir::Operation* op,
                                      std::vector<Type>* out_types,
                                      std::vector<std::vector<int>>* out_shapes,
                                      const OpLoweringGroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));
    auto out_shape = ::common::vectorize<int>(type_info.dims());
    if (out_shape.empty()) {
      out_shape.push_back(1);
    }
    out_shapes->push_back(std::move(out_shape));
  }
}

void OpLowererImpl::CollectOutputInfo(
    ::pir::Operation* op,
    std::vector<Type>* out_types,
    std::vector<std::vector<ir::Dim>>* out_shapes,
    const OpLoweringGroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));

    auto ForEachDimExpr = [&](const auto& DoEach) {
      const auto& dims = type_info.dims();
      if (::common::contain_unknown_dim(dims)) {  // dynamic shape
        const auto& sym_vec = group->GetShapeOrDataExprs(out_value).shape();
        std::vector<ir::Dim> sym_shape;
        for (const auto& sym : sym_vec) {
          DoEach(sym);
        }
      } else {  // static shape
        auto out_shape = ::common::vectorize<int64_t>(dims);
        for (int64_t dim : out_shape) {
          DoEach(symbol::DimExpr{dim});
        }
      }
    };
    std::vector<ir::Dim> sym_shape;
    ForEachDimExpr(
        [&](const auto& sym) { sym_shape.emplace_back(output_id, sym); });
    if (sym_shape.empty()) {
      sym_shape.emplace_back(output_id, symbol::DimExpr{1});
    }
    out_shapes->emplace_back(std::move(sym_shape));
  }
}

std::string OpLowererImpl::ValueName(::pir::Value value) {
  auto name = name_gene_->GetOrNew(value, CompatibleInfo::kNamePrefix);

  return name;
}

common::Type OpLowererImpl::GetTensorDtype(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == ValueName(iter.first)) {
      return GetTensorDtype(iter.first);
    }
  }
  VLOG(4) << name << " is not in tensor map, return FP32 by default.";
  return common::F32();
}

common::Type OpLowererImpl::GetTensorDtype(const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto in_shape = ::common::vectorize<int>(type_info.dims());
  auto dtype = type_info.dtype();
  return CompatibleInfo::ConvertIRType(dtype);
}

bool OpLowererImpl::IsInTensorMap(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == ValueName(iter.first)) {
      return true;
    }
  }
  return false;
}

ir::LoweredFunc OpLowererImpl::GenerateInferShapeFunc(
    const OpLoweringGroupPtr& group,
    const std::vector<ir::Tensor> group_func_arg_tensors,
    const std::vector<ir::Argument> group_func_args) {
  // CHECK_EQ(group_func_arg_tensors.size(), group_func_args.size());
  std::vector<ir::Expr> ir_bodys;
  int output_tensor_idx = 0;
  for (int tensor_arg_idx = 0; tensor_arg_idx < group_func_arg_tensors.size();
       ++tensor_arg_idx) {
    auto tensor_dim = group_func_arg_tensors[tensor_arg_idx]->sym_shape;
    int tensor_dim_size = tensor_dim.size();
    auto tensor_shape = group_func_arg_tensors[tensor_arg_idx]->shape;

    ir::Var tensor_shape_args(TENSOR_SHAPE_ARGS, type_of<int64_t**>());
    for (int i = 0; i < tensor_shape.size(); i++) {
      ir::Expr call_set_infer_shape_value =
          ir::Call::Make(type_of<void>(),
                         runtime::intrinsic::infer_shape_set_value,
                         {ir::Expr(output_tensor_idx),
                          ir::Expr(i),
                          tensor_shape[i],
                          tensor_shape_args},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir_bodys.push_back(call_set_infer_shape_value);
    }
    ++output_tensor_idx;
  }
  ir::LoweredFunc infer_shape_func =
      ir::_LoweredFunc_::Make(group->FuncName() + "_infer_shape",
                              group_func_args,
                              ir::Block::Make(ir_bodys),
                              {});
  return infer_shape_func;
}
ir::Expr OpLowererImpl::LowerX86(const OpLoweringGroupPtr& group,
                                 const std::vector<::pir::Operation*>& ops,
                                 bool apply_op_schedule) {
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;

  auto need_lower_x86 = [&]() -> bool {
    for (auto* op : ops) {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        auto in = op->operand_source(i);
        if (!in || !in.type()) {
          continue;
        }
        auto type_info = in.type().dyn_cast<paddle::dialect::DenseTensorType>();
        auto dtype = type_info.dtype();
        const auto& dims = type_info.dims();
        std::vector<ir::Dim> sym_shape;
        // 1. dynamic shape not need lower x86
        if (::common::contain_unknown_dim(dims)) {
          return false;
        }
        // 2. size < 4 not need lower x86
        int64_t sym_shape_size = 1;
        for (int i = 0; i < dims.size(); ++i) {
          sym_shape_size *= dims[i];
          if (sym_shape_size > 4) {
            return false;
          }
        }
      }

      std::vector<Type> out_types;
      std::vector<std::vector<ir::Dim>> out_shapes;
      CollectOutputInfo(op, &out_types, &out_shapes, group);
      for (const auto& tt : out_types) {
        // 3. float16 not need lower x86
        if (tt.is_float16()) {
          return false;
        }
      }
    }
    return true;
  };
  if (!need_lower_x86()) {
    return ir::Expr(-1);
  }

  this->target_ = common::DefaultHostTarget();
  cinn::runtime::CurrentTarget::SetCurrentTarget(this->target_);

  std::vector<ir::Expr> func_bodies =
      LowerOps(group,
               ops,
               apply_op_schedule,
               &OpLowererImpl::DyShapeScheduleDetermineFunction,
               &group_func_arg_tensors,
               &tensor_map,
               &tmp_tensor_info);
  this->target_ = common::DefaultNVGPUTarget();
  cinn::runtime::CurrentTarget::SetCurrentTarget(this->target_);
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir_sch.MergeExprs();
  auto X86Expr = ir::ir_utils::IRCopy(ir_sch.GetModule().GetExprs().at(0));
  return X86Expr;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
