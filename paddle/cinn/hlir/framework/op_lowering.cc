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

#include "cinn/hlir/framework/op_lowering.h"

#include "cinn/hlir/framework/op_lowering_util.h"
#include "cinn/hlir/op/external_api_registry.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/transform_gpu_forloop.h"

DECLARE_bool(cinn_ir_schedule);
DECLARE_bool(cinn_use_cuda_vectorize);

namespace cinn {
namespace hlir {
namespace framework {

using common::bfloat16;
using common::float16;

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;
using framework::StrategyFunction;

using common::GraphEdge;
using common::GraphNode;
using common::Type;
using namespace lang;

using Comparator = Graph::Group::SharedGroupComparator;
using Hasher = Graph::Group::SharedGroupHasher;
using cinn::hlir::op::ExternalApiRegistry;

OpLowerer::OpLowerer(
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const Target& target)
    : type_dict_(type_dict), shape_dict_(shape_dict), target_(target) {}

std::vector<ir::LoweredFunc> OpLowerer::Lower(GroupPtr& group) {
  VLOG(3) << "Lowering Group : " << group->group_id
          << " , Op Pattern : " << group->op_pattern_kind;
  group->input_names.clear();
  group->output_names.clear();
  if (FLAGS_cinn_ir_schedule) {
    switch (group->op_pattern_kind) {
      case framework::kElementWise:
      case framework::kBroadcast:
      case framework::kInjective:
        return IRLowerOp(&OpLowerer::IRElementwiseCompute,
                         &OpLowerer::IRElementwiseSchedule,
                         group);
      case framework::kReduction:
        return IRLowerOp(
            &OpLowerer::IRReduceCompute, &OpLowerer::IRReduceSchedule, group);
      case framework::kOutFusible:
        LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
      case framework::kNonFusible:
        return IRLowerNonFusibleOp(group, /*apply_impl_schedule = */ true);
      default:
        LOG(FATAL) << "Group Pattern Kind Is Unknown!";
    }
  } else {
    LOG(FATAL) << "Previous IR Schedule Is Not Implemented!";
  }
}

std::vector<ir::LoweredFunc> OpLowerer::LowerWithoutSchedule(GroupPtr& group) {
  VLOG(3) << "Lowering Group : " << group->group_id
          << " , Op Pattern : " << group->op_pattern_kind;
  if (FLAGS_cinn_ir_schedule) {
    switch (group->op_pattern_kind) {
      case framework::kElementWise:
      case framework::kBroadcast:
      case framework::kInjective:
        return IRLowerOpWithoutSchedule(&OpLowerer::IRElementwiseCompute,
                                        group);
      case framework::kReduction:
        return IRLowerOpWithoutSchedule(&OpLowerer::IRReduceCompute, group);
      case framework::kOutFusible:
        LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
      case framework::kNonFusible:
        return IRLowerNonFusibleOp(group, /*apply_impl_schedule = */ false);
      default:
        LOG(FATAL) << "Group Pattern Kind kNonFusible Is Not Implemented!";
    }
  } else {
    LOG(FATAL) << "Previous IR Schedule Is Not Implemented!";
  }
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOp(IRComputeFunction compute,
                                                  IRScheduleFunction schedule,
                                                  GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  // do compute.
  VLOG(3) << "group->fused_sub_groups.size() is : "
          << group->fused_sub_groups.size();
  std::vector<Expr> ast_exprs;
  if (group->fused_sub_groups.size() == 0) {
    ast_exprs = (this->*compute)(stages,
                                 arg_tensors,
                                 tensor_map,
                                 group,
                                 group,
                                 /*apply_impl_schedule = */ true);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      auto exprs = (this->*compute)(stages,
                                    arg_tensors,
                                    tensor_map,
                                    group,
                                    sub_group,
                                    /*apply_impl_schedule = */ true);
      ast_exprs.insert(ast_exprs.end(), exprs.begin(), exprs.end());
    }
  }
  ir::ModuleExpr mod_expr(ast_exprs);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  Node* first = nullptr;
  Node* second = nullptr;

  VLOG(3) << "Before IRLowerOp schedule, ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
  // do schedule.
  IRSchedule(ir_sch, group, tensor_map);
  VLOG(3) << "After IRLowerOp schedule, ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
  // function args
  group->input_names.clear();
  std::vector<ir::Argument> func_args;
  for (auto& args : arg_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
    // input args
    func_args.emplace_back(args->buffer, ir::Argument::IO::kInput);
  }

  group->output_names.clear();
  for (auto& node : group->output_nodes) {
    // output node data name.
    for (auto node_data : GetAllNodeData(node)) {
      group->output_names.push_back(node_data->id());
    }
    // collect all output tensor.
    std::string post = "";
    std::string prefix = GetNodeData(node)->id();
    for (int idx = 0; idx < 1; ++idx) {
      CHECK(tensor_map.count(prefix)) << "Can't find output tensor " << prefix;
      if (!tensor_map.count(prefix + post)) {
        break;
      }
      auto tensor = tensor_map[prefix + post];
      arg_tensors.push_back(tensor);
      // output args
      func_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      // update post
      post = "_" + std::to_string(idx);
    }
  }
  auto func_body = ir_sch.GetModule().GetExprs().at(0);
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  auto temp_buffers = lang::GetTempBuffers(arg_tensors, stages, func_body);
  auto func = ir::_LoweredFunc_::Make(group->GetFuncName(),
                                      func_args,
                                      ir_sch.GetModule().GetExprs().at(0),
                                      temp_buffers);
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
  return {func};
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerOpWithoutSchedule(
    IRComputeFunction compute, GroupPtr& group) {
  poly::StageMap stages;
  std::vector<ir::Tensor> arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  // do compute.
  VLOG(3) << "group->fused_sub_groups.size() is : "
          << group->fused_sub_groups.size();
  std::vector<Expr> ast_exprs;
  if (group->fused_sub_groups.size() == 0) {
    ast_exprs = (this->*compute)(stages,
                                 arg_tensors,
                                 tensor_map,
                                 group,
                                 group,
                                 /*apply_impl_schedule = */ false);
  } else {
    for (auto& sub_group : group->fused_sub_groups) {
      auto exprs = (this->*compute)(stages,
                                    arg_tensors,
                                    tensor_map,
                                    group,
                                    sub_group,
                                    /*apply_impl_schedule = */ false);
      ast_exprs.insert(ast_exprs.end(), exprs.begin(), exprs.end());
    }
  }
  ir::ModuleExpr mod_expr(ast_exprs);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();

  VLOG(3) << "After IRLowerOp compute, ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
  // function args
  group->input_names.clear();
  std::vector<ir::Argument> func_args;
  for (auto& args : arg_tensors) {
    // input node data name.
    group->input_names.push_back(args->name);
    // input args
    func_args.emplace_back(args->buffer, ir::Argument::IO::kInput);
  }

  group->output_names.clear();
  for (auto& node : group->output_nodes) {
    // output node data name.
    for (auto node_data : GetAllNodeData(node)) {
      group->output_names.push_back(node_data->id());
    }
    // collect all output tensor.
    std::string post = "";
    std::string prefix = GetNodeData(node)->id();
    for (int idx = 0; idx < 1; ++idx) {
      CHECK(tensor_map.count(prefix)) << "Can't find output tensor " << prefix;
      if (!tensor_map.count(prefix + post)) {
        break;
      }
      auto tensor = tensor_map[prefix + post];
      arg_tensors.push_back(tensor);
      // output args
      func_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      // update post
      post = "_" + std::to_string(idx);
    }
  }

  std::unordered_set<std::string> args_map;
  for (auto arg : func_args) {
    args_map.insert(arg.name());
  }

  for (auto& tensor : tensor_map) {
    if (args_map.count("_" + tensor.first)) {
      continue;
    }
    arg_tensors.push_back(tensor.second);
    // use the underlying tensor name to be consistent with the argument name in
    // the lowered function
    group->output_names.push_back(tensor.second->name);
    func_args.emplace_back(tensor.second->buffer, ir::Argument::IO::kOutput);
  }

  auto func_body = ir_sch.GetModule().GetExprs().at(0);
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&(func_body));
#endif

  auto temp_buffers = lang::GetTempBuffers(arg_tensors, stages, func_body);
  auto func = ir::_LoweredFunc_::Make(group->GetFuncName(),
                                      func_args,
                                      ir_sch.GetModule().GetExprs().at(0),
                                      temp_buffers);
  func->PrepareBufferCastExprs();
  func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();

  return {func};
}

std::vector<Expr> OpLowerer::IRElementwiseCompute(
    poly::StageMap& stages,
    std::vector<ir::Tensor>& func_tensors,
    std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const GroupPtr& group,
    const GroupPtr& sub_group,
    bool apply_impl_schedule) {
  VLOG(2) << "ElementwiseCompute Group : " << sub_group->group_id;
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    VLOG(4) << "Lower op: " << node->op()->name;
    auto node_data = GetNodeData(node);
    CHECK_EQ(GetAllNodeData(node).size(), 1U);
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(
        node, func_tensors, tensor_map, this->type_dict_, this->shape_dict_));
    for (auto& tensor : tensor_inputs) {
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }
    // set tensor name = node data name
    cinn_inputs.push_back(common::CINNValue(node_data->id()));

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
    auto impl = OpStrategy::SelectImpl(strategy[node->op()](
        node->attrs, tensor_inputs, out_types, out_shapes, this->target_));
    // do compute
    common::CINNValuePack pack =
        impl->fcompute(common::CINNValuePack{cinn_inputs});
    CHECK_EQ(pack.size(), 2U);

    Expr expr = pack[0];
    poly::StageMap node_stages = pack.back();
    tensor_inputs.push_back(expr.as_tensor_ref());
    tensor_map[node_data->id()] = expr.as_tensor_ref();

    auto func = lang::LowerVec("fn_" + node->id(),
                               node_stages,
                               tensor_inputs,
                               {},
                               {},
                               nullptr,
                               this->target_,
                               true);
    CHECK_EQ(func.size(), 1);

    if (apply_impl_schedule) {
      std::vector<common::CINNValue> schedule_inputs;
      // collect tensor
      for (int idx = 0; idx < pack.size() - 1; ++idx) {
        CHECK(pack[idx].is_tensor());
        schedule_inputs.push_back(common::CINNValue(pack[idx]));
      }
      for (auto& f : func) {
        schedule_inputs.push_back(common::CINNValue(f->body));
      }
      // do ast tree schedule
      common::CINNValuePack expr_pack =
          impl->fschedule(common::CINNValuePack{schedule_inputs});

      CHECK_EQ(expr_pack.size(), 1);
      Expr ast_expr = expr_pack[0];
      ast_exprs.push_back(ast_expr);
    } else {
      ast_exprs.push_back(func[0]->body);
    }
  }

  return ast_exprs;
}

void OpLowerer::IRElementwiseSchedule(
    ir::IRSchedule& ir_sch,
    std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const GroupPtr& group,
    const GroupPtr& sub_group,
    Node*&,
    Node*&) {
  VLOG(2) << "IRElementwiseSchedule Group : " << sub_group->group_id;
  auto master_node = *group->master_nodes.begin();
  auto manster_tensor = tensor_map[GetNodeData(master_node)->id()];

  for (int idx = sub_group->nodes.size() - 1; idx >= 0; --idx) {
    auto node = sub_group->nodes[idx];
    auto node_tensor = tensor_map[GetNodeData(node)->id()];

    VLOG(3) << "Schedule node -> " << node->id()
            << " var : " << node_tensor->name;
    if (group->master_nodes.count(node)) {
      continue;
    }

    if (IsConstOp(node) && !group->output_nodes.count(node)) {
      ir_sch.ComputeInline(ir_sch.GetBlock(node_tensor->name));
      continue;
    }

    // if node is fringe node or internal node, fringe node is output node of
    // sub-graph
    if (group->output_nodes.count(node) || group->internal_nodes.count(node) ||
        sub_group->internal_nodes.count(node)) {
      // internal node use buffer
      if (!group->output_nodes.count(node)) {
        auto node_block = ir_sch.GetBlock(node_tensor->name);
        ir_sch.SetBuffer(node_block, "local", true);
      }

      auto node_block = ir_sch.GetBlock(node_tensor->name);
      auto master_loops = ir_sch.GetLoops(manster_tensor->name);
      ir_sch.SimpleComputeAt(node_block, master_loops.back());
      continue;
    }

    // others elemenwise internal node use compute-inline
    ir_sch.ComputeInline(ir_sch.GetBlock(node_tensor->name));
  }
}

std::vector<Expr> OpLowerer::IRReduceCompute(
    poly::StageMap& stages,
    std::vector<ir::Tensor>& func_args,
    std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const GroupPtr& group,
    const GroupPtr& sub_group,
    bool apply_impl_schedule) {
  VLOG(2) << "ReduceCompute Group : " << sub_group->group_id;
  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  std::vector<Expr> ast_exprs;
  for (auto& node : sub_group->nodes) {
    auto node_data = GetNodeData(node);
    VLOG(3) << "In ReduceCompute, process node: " << node->id()
            << " with op type: " << node->op()->name;

    std::vector<common::CINNValue> cinn_inputs;
    std::vector<ir::Tensor> tensor_inputs = std::move(CollectInputTensor(
        node, func_args, tensor_map, this->type_dict_, this->shape_dict_));
    for (auto& tensor : tensor_inputs) {
      cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
    }
    cinn_inputs.push_back(common::CINNValue(node_data->id()));

    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));

    auto impl = OpStrategy::SelectImpl(cinn_strategy[node->op()](
        node->attrs, tensor_inputs, out_types, out_shapes, target_));
    // do compute
    common::CINNValuePack pack =
        impl->fcompute(common::CINNValuePack{cinn_inputs});

    CHECK_GE(pack.size(), 2UL);
    CHECK_LE(pack.size(), 5UL);
    poly::StageMap tmp_stages = pack.back();

    std::string post = "";
    for (int idx = 0; idx < pack.size() - 1; ++idx) {
      Expr expr = pack[idx];
      tensor_map[node_data->id() + post] = expr.as_tensor_ref();
      // As op may has more than 1 output tensor, using id + "_0"/"_1" as key.
      post = "_" + std::to_string(idx);

      // Insert outout tensors
      if (!expr.as_tensor_ref()->buffer.defined() ||
          this->target_ != common::DefaultNVGPUTarget()) {
        tensor_inputs.push_back(expr.as_tensor_ref());
      }
    }
    auto func = lang::LowerVec("fn_" + node->id(),
                               tmp_stages,
                               tensor_inputs,
                               {},
                               {},
                               nullptr,
                               this->target_,
                               true);

    // node is kReduction
    if (op_pattern_dict[node->op()] == framework::kReduction &&
        apply_impl_schedule) {
      std::vector<common::CINNValue> schedule_inputs;
      // collect tensor
      for (int idx = 0; idx < pack.size() - 1; ++idx) {
        CHECK(pack[idx].is_tensor());
        schedule_inputs.push_back(common::CINNValue(pack[idx]));
      }
      for (auto& f : func) {
        schedule_inputs.push_back(common::CINNValue(f->body));
      }
      // do ast tree schedule
      common::CINNValuePack expr_pack =
          impl->fschedule(common::CINNValuePack{schedule_inputs});
      // ast tree after schedule.
      Expr ast_expr = expr_pack[0];
      ast_exprs.push_back(ast_expr);
    } else if (group->master_nodes.count(node)) {
      // as master node should copy transform from reducer, left it to reduce
      // schedule.
      ast_exprs.push_back(func[0]->body);
    } else {
      ast_exprs.push_back(func[0]->body);
    }
  }

  return ast_exprs;
}

void OpLowerer::IRReduceSchedule(
    ir::IRSchedule& ir_sch,
    std::unordered_map<std::string, ir::Tensor>& tensor_map,
    const GroupPtr& group,
    const GroupPtr& sub_group,
    Node*& master,
    Node*& reducer) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto OrderAssignReduce = [this](ir::IRSchedule& ir_sch,
                                  const std::string& block_name,
                                  const std::vector<int>& axes,
                                  const bool just_reorder = false) {
    // reorder none-last reduce axis to last.
    // like: shape = [16,16,16,16,16],axes = [1,3] -> new order = [0, 2, 4, 1,
    // 3].
    std::vector<int> order;
    int n_out_dims = ir_sch.GetLoops(block_name).size();
    for (int idx = 0; idx < n_out_dims; ++idx) {
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        order.push_back(idx);
      }
    }
    for (auto axis : axes) {
      order.push_back(axis);
    }
    ir_sch.Reorder(ir_sch.GetBlock(block_name), order);

    if (just_reorder) {
      return;
    }
    // fuse others none-reduce axis.
    int last_dimension_num = n_out_dims - axes.back() - 1;
    int index = n_out_dims - last_dimension_num - axes.size();

    // fuse last_dimension_num - 1 times
    for (auto idx = index; idx < index + last_dimension_num - 1; ++idx) {
      ir_sch.Fuse(block_name, {index, index + 1});
    }

    auto loops = ir_sch.GetLoops(block_name);
    auto psize = ir::GetLoopExtent(loops[index]);
    if (psize > this->target_.max_num_threads()) {
      for (int idx = this->target_.max_num_threads(); idx > 0; --idx) {
        if (psize % idx == 0) {
          ir_sch.Split(loops[index], {-1, idx});
          break;
        }
        CHECK_GT(idx, 1);
      }
    }

    // fuse index - 1 times
    for (int idx = 0; idx < index - 1; ++idx) {
      ir_sch.Fuse(block_name, {0, 1});
    }
  };

  auto WithoutLastDimInReduce = [](const std::vector<int>& inshape,
                                   std::vector<int>& axes) {
    // if last axis is in reduce.
    axes = axes.empty() ? inshape : axes;
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx];
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  };

  auto ScheduleAssignReduceWithoutLast = [this, OrderAssignReduce](
                                             ir::IRSchedule& ir_sch,
                                             const std::string& block_name,
                                             const std::vector<int>& inshape,
                                             std::vector<int>& axes) {
    axes = axes.empty() ? inshape : axes;
    int lane = 1;
    int max_num_threads = this->target_.max_num_threads();
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      lane *= inshape[idx];
    }
    CHECK_LE(lane, max_num_threads / 2)
        << "Parallel threads must less equal max_num_threads/2 on gpu!";
    int pos = 0;
    int index = axes.size() - 1;
    for (; index >= 0; --index) {
      if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
        pos = axes[index + 1];
        break;
      }

      lane *= inshape[axes[index]];
      if (lane > max_num_threads / 2) {
        pos = axes[index];
        break;
      }

      if (index == 0) {
        pos = axes[0];
      }
    }

    if (lane > max_num_threads / 2) {
      int prefix = inshape[axes[index]];
      int tail = lane / prefix;
      for (int idx = max_num_threads / tail; idx > (max_num_threads / 2) / tail;
           --idx) {
        if (prefix % idx == 0) {
          ir_sch.Split(block_name, axes[index], {-1, idx});
          break;
        }
        CHECK_GT(idx - 1, (max_num_threads / 2) / tail)
            << "idx should greater than (max_num_threads / 2) / tail.";
      }
    }

    // insert 1
    for (int idx = 0; idx < axes.size() - 1 - index; ++idx) {
      auto loops = ir_sch.GetLoops(block_name);
      ir_sch.Split(block_name, pos, {-1, ir::GetLoopExtent(loops[pos])});
    }
    OrderAssignReduce(ir_sch, block_name, axes);
    // return insert 1
    int start_index = ir_sch.GetLoops(block_name).size() - axes.size();
    for (int idx = 0; idx < axes.size(); ++idx) {
      auto loops = ir_sch.GetLoops(block_name);
      if (ir::GetLoopExtent(loops[start_index]) == 1) {
        ir_sch.Fuse({loops[start_index - 1], loops[start_index]});
      } else {
        ++start_index;
      }
    }
  };

  auto ScheduleAssignReduceWithLast = [this, OrderAssignReduce](
                                          ir::IRSchedule& ir_sch,
                                          const std::string& block_name,
                                          const std::vector<int>& inshape,
                                          std::vector<int>& axes) {
    // find first reduce and second reduce axis.
    axes = axes.empty() ? inshape : axes;
    int lane = 1;
    int index = static_cast<int>(axes.size()) - 1;
    auto max_num_threads = this->target_.max_num_threads();
    for (; index >= 0; --index) {
      if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
        break;
      }
      lane *= inshape[axes[index]];
      if (index == 0 && lane <= max_num_threads) {
        LOG(FATAL)
            << "Error! lane is less equal than max_num_threads, Please check!";
      }
      if (lane >= max_num_threads / 2) {
        if (lane <= max_num_threads) {
          --index;
        }
        break;
      }
    }
    std::vector<int> first_axes(axes.begin(), axes.begin() + index + 1);
    if (lane > max_num_threads) {
      // last reduce axis size > 1024
      if (index == static_cast<int>(axes.size()) - 1) {
        int idx = max_num_threads;
        do {
          if (lane % idx == 0) {
            ir_sch.Split(block_name, axes[index], {-1, idx});
            break;
          }
          --idx;
        } while (idx >= max_num_threads / 2);
        // if can't be divide by(1024, 512), it's shouldn't be fused.
        CHECK_GE(idx, max_num_threads / 2) << "Check bounds exist, can't fuse!";
      } else {
        int axis = axes[index];
        int prefix = inshape[axis];
        int tail = lane / prefix;
        for (int idx = max_num_threads / tail;
             idx > (max_num_threads / 2) / tail;
             --idx) {
          if (prefix % idx == 0) {
            ir_sch.Split(block_name, axis, {-1, idx});
            break;
          }
          CHECK_GT(idx, (max_num_threads / 2) / tail)
              << "Error, it's shouldn't fuse!";
        }
      }
      OrderAssignReduce(ir_sch, block_name, first_axes);
    } else {
      int fuse_times = axes.size() - (index + 1) - 1;
      for (int idx = 0; idx < fuse_times; ++idx) {
        ir_sch.Fuse(block_name, {axes[index + 1], axes[index + 1] + 1});
      }
      OrderAssignReduce(ir_sch, block_name, first_axes, true);
      // fuse axis before reduce to bind blockidx.
      for (int idx = 0; idx < (inshape.size() - axes.size()) - 1; ++idx) {
        ir_sch.Fuse(block_name, {0, 1});
      }
    }
  };

  if (master == nullptr && reducer == nullptr) {
    auto blocks = ir_sch.GetAllBlocks();
    for (int idx = blocks.size() - 1; idx >= 0; --idx) {
      auto block = blocks[idx];
      CHECK(block->as<ir::ScheduleBlockRealize>());
      CHECK(block->as<ir::ScheduleBlockRealize>()
                ->schedule_block->as<ir::ScheduleBlock>());
      if (!tensor_map.count(block->as<ir::ScheduleBlockRealize>()
                                ->schedule_block->as<ir::ScheduleBlock>()
                                ->name)) {
        continue;
      }

      for (auto node : group->master_nodes) {
        if (GetNodeData(node)->id() ==
            block->as<ir::ScheduleBlockRealize>()
                ->schedule_block->as<ir::ScheduleBlock>()
                ->name) {
          if (op_pattern_dict[node->op()] != framework::kReduction) {
            master = node;
            break;
          }

          if (op_pattern_dict[node->op()] == framework::kReduction && master) {
            reducer = node;
            break;
          }
        }
      }

      if (master && reducer) {
        break;
      }
    }
    CHECK((master && reducer) || (!master && !reducer))
        << "Can't find Master reducer!";
    if (!master && !reducer) {
      master = *group->master_nodes.begin();
      reducer = *group->master_nodes.begin();
    }

    // do master schedule.
    if (op_pattern_dict[master->op()] != framework::kReduction) {
      VLOG(2) << "Do Master Schedule : " << master->id();
      auto master_data = GetNodeData(master);
      CHECK(master_data);
      CHECK(tensor_map.count(master_data->id()));
      auto master_tensor = tensor_map[master_data->id()];
      auto loops = ir_sch.GetLoops(master_tensor->name);
      if (op_pattern_dict[master->op()] == framework::kElementWise) {
        ir_sch.FlattenLoops(loops, true);
      } else {
        ir_sch.FlattenLoops(loops, false);
      }

      auto reducer_data = GetNodeData(reducer);
      auto reducer_tensor = tensor_map[reducer_data->id()];
      auto rloops = ir_sch.GetLoops(reducer_tensor->name);

      // assign master loops to reducer loops without reduce axis.
      int extend = 1;
      std::vector<int> factors;
      auto sloops = ir_sch.GetLoops(master_tensor->name);
      for (auto& loop : rloops) {
        // without last reduce axis, so check loop extend.
        extend *= loop.As<ir::For>()->extent.as_int32();
        if (extend > sloops.back().As<ir::For>()->extent.as_int32()) {
          break;
        }
        CHECK_LE(extend, sloops.back().As<ir::For>()->extent.as_int32());
        factors.push_back(loop.As<ir::For>()->extent.as_int32());
      }
      ir_sch.Split(sloops.back(), factors);

      auto nloops = ir_sch.GetLoops(master_tensor->name);
      CHECK_GE(rloops.size(), nloops.size());
      for (int idx = 0; idx < nloops.size(); ++idx) {
        nloops[idx].As<ir::For>()->set_bind_info(
            rloops[idx].As<ir::For>()->bind_info());
      }
    }
    // do reducer schedule.
    {
      auto reducer_data = GetNodeData(reducer);
      auto reducer_tensor = tensor_map[reducer_data->id()];
      CHECK(reducer->attrs.attr_store.count("dim"));
      auto reducer_axes =
          absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
      CHECK(reducer->inlinks_in_order().size());
      CHECK(this->shape_dict_.count(
          reducer->inlinks_in_order()[0]->source()->id()));
      auto reducer_shape =
          this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());

      if (reducer_axes.empty()) {
        for (int i = 0; i < reducer_shape.size(); ++i) {
          reducer_axes.emplace_back(i);
        }
      }

      bool without_last_dim =
          WithoutLastDimInReduce(reducer_shape, reducer_axes);

      std::unordered_set<Node*> visited_nodes;
      for (auto node : group->master_nodes) {
        VLOG(2) << "Schedule reduce node -> " << node->id();
        if (op_pattern_dict[node->op()] != framework::kReduction) {
          continue;
        }
        auto node_data = GetNodeData(node);
        auto node_tensor = tensor_map[node_data->id()];

        if (!group->output_nodes.count(node)) {
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SetBuffer(node_block, "local", true);
        }
        if (node == reducer) {
          continue;
        }
        auto node_shape =
            this->shape_dict_.at(node->inlinks_in_order()[0]->source()->id());
        if (without_last_dim) {
          VLOG(2) << "Reduce Schedule WithoutLastDimInReduce";
          // find a shape to do simple compute at.
          auto tmp_reducer = reducer;
          auto tmp_reducer_shape = reducer_shape;
          if (node_shape != reducer_shape) {
            // try to find the same shape reduce from visited_nodes
            for (auto visited : visited_nodes) {
              auto shape = this->shape_dict_.at(
                  visited->inlinks_in_order()[0]->source()->id());
              if (shape == node_shape) {
                tmp_reducer = visited;
                tmp_reducer_shape = shape;
                break;
              }
            }
          }
          visited_nodes.insert(node);
          auto tmp_reducer_data = GetNodeData(tmp_reducer);
          auto tmp_reducer_tensor = tensor_map[tmp_reducer_data->id()];

          // using block shuffle reduce.
          if (tensor_map.count(reducer_data->id() + "_1")) {
            auto node_0_tensor = tensor_map[node_data->id() + "_0"];
            auto node_0_block = ir_sch.GetBlock(node_0_tensor->name);

            auto tmp_reducer_0_tensor =
                tensor_map[tmp_reducer_data->id() + "_0"];
            auto tmp_reducer_0_loops =
                ir_sch.GetLoops(tmp_reducer_0_tensor->name);

            if (tmp_reducer_shape == node_shape) {
              ir_sch.SimpleComputeAt(node_0_block, tmp_reducer_0_loops.back());
              // init compute at reduce
              int loop_depth =
                  ir_sch.GetLoops(node_0_tensor->name + "__reduce_init").size();
              ir_sch.SimpleComputeAt(
                  ir_sch.GetBlock(node_0_tensor->name + "__reduce_init"),
                  ir_sch.GetLoops(node_0_tensor->name)[loop_depth - 1]);
            } else {
              if (tmp_reducer_0_tensor->shape.back() ==
                  node_0_tensor->shape.back()) {
                int num_reduce_axis = tmp_reducer_0_tensor->reduce_axis.size();
                CHECK_GE(static_cast<int>(tmp_reducer_0_loops.size()) -
                             num_reduce_axis - 1,
                         0);
                ir_sch.SimpleComputeAt(
                    node_0_block,
                    tmp_reducer_0_loops[tmp_reducer_0_loops.size() -
                                        num_reduce_axis - 1]);
                // init compute at reduce
                int loop_depth =
                    ir_sch.GetLoops(node_0_tensor->name + "__reduce_init")
                        .size();
                ir_sch.SimpleComputeAt(
                    ir_sch.GetBlock(node_0_tensor->name + "__reduce_init"),
                    ir_sch.GetLoops(node_0_tensor->name)[loop_depth - 1]);
              } else {
                CHECK_GE(static_cast<int>(tmp_reducer_0_loops.size()), 2);
                ir_sch.SimpleComputeAt(node_0_block, tmp_reducer_0_loops[0]);
              }
            }
            ir_sch.SimpleComputeAt(
                ir_sch.GetBlock(node_tensor->name),
                ir_sch.GetLoops(tmp_reducer_tensor->name).back());
          } else {
            if (tmp_reducer_shape == node_shape) {
              ir_sch.SimpleComputeAt(
                  ir_sch.GetBlock(node_tensor->name),
                  ir_sch.GetLoops(tmp_reducer_tensor->name).back());
            } else {
              int num_reduce_axis = tmp_reducer_tensor->reduce_axis.size();
              auto tmp_reducer_loops =
                  ir_sch.GetLoops(tmp_reducer_tensor->name);
              CHECK_GE(static_cast<int>(tmp_reducer_loops.size()) -
                           num_reduce_axis - 1,
                       0);
              ir_sch.SimpleComputeAt(
                  ir_sch.GetBlock(node_tensor->name),
                  tmp_reducer_loops[tmp_reducer_loops.size() - num_reduce_axis -
                                    1]);
            }
            // init compute at reduce
            int loop_depth =
                ir_sch.GetLoops(node_tensor->name + "__reduce_init").size();
            ir_sch.SimpleComputeAt(
                ir_sch.GetBlock(node_tensor->name + "__reduce_init"),
                ir_sch.GetLoops(node_tensor->name)[loop_depth - 1]);
          }
        } else {
          VLOG(2) << "Reduce Schedule WithLastDimInReduce";
          // if with column reduce behind.
          if (tensor_map.count(node_data->id() + "_1")) {
            auto reducer_1_tensor = tensor_map[reducer_data->id() + "_1"];
            auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];

            auto node_1_tensor = tensor_map[node_data->id() + "_1"];
            auto node_0_tensor = tensor_map[node_data->id() + "_0"];

            auto node_block_1 = ir_sch.GetBlock(node_1_tensor->name);
            auto node_block_0 = ir_sch.GetBlock(node_0_tensor->name);
            auto node_block = ir_sch.GetBlock(node_tensor->name);

            ir_sch.SimpleComputeAt(
                node_block, ir_sch.GetLoops(reducer_tensor->name).back());
            ir_sch.SimpleComputeAt(
                node_block_0, ir_sch.GetLoops(reducer_0_tensor->name).back());
            ir_sch.SimpleComputeAt(
                node_block_1, ir_sch.GetLoops(reducer_1_tensor->name).back());
            // init compute at reduce
            int loop_depth =
                ir_sch.GetLoops(node_1_tensor->name + "__reduce_init").size();
            ir_sch.SimpleComputeAt(
                ir_sch.GetBlock(node_1_tensor->name + "__reduce_init"),
                ir_sch.GetLoops(node_1_tensor->name)[loop_depth - 1]);
          } else if (tensor_map.count(node_data->id() + "_0")) {
            auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];
            auto node_0_tensor = tensor_map[node_data->id() + "_0"];

            auto node_0_block = ir_sch.GetBlock(node_0_tensor->name);
            auto node_block = ir_sch.GetBlock(node_tensor->name);
            ir_sch.SimpleComputeAt(
                node_block, ir_sch.GetLoops(reducer_tensor->name).back());
            ir_sch.SimpleComputeAt(
                node_0_block, ir_sch.GetLoops(reducer_0_tensor->name).back());
          } else {
            LOG(FATAL) << "Error! Unkown Reduce Type, Please Check!";
          }
        }
      }

      if (without_last_dim) {
        if (tensor_map.count(reducer_data->id() + "_1")) {
          auto reducer_tensor = tensor_map[GetNodeData(reducer)->id()];
          auto reducer_loops = ir_sch.GetLoops(reducer_tensor->name);
          ir_sch.SyncThreads(reducer_loops[0], false);
        }
      }
    }
  }

  // master node
  auto master_data = GetNodeData(master);
  CHECK(master_data);
  CHECK(tensor_map.count(master_data->id()));
  auto master_tensor = tensor_map[master_data->id()];
  auto master_shape = this->shape_dict_.at(master_data->id());
  auto master_size = std::accumulate(
      master_shape.begin(), master_shape.end(), 1, std::multiplies<int>());

  // reducer node
  auto reducer_data = GetNodeData(reducer);
  CHECK(reducer_data);
  CHECK(reducer->inlinks_in_order().size());
  CHECK(
      this->shape_dict_.count(reducer->inlinks_in_order()[0]->source()->id()));
  auto reducer_shape =
      this->shape_dict_.at(reducer->inlinks_in_order()[0]->source()->id());
  auto reduce_size = std::accumulate(
      reducer_shape.begin(), reducer_shape.end(), 1, std::multiplies<int>());

  CHECK(reducer->attrs.attr_store.count("dim"));
  auto reducer_axes =
      absl::get<std::vector<int>>(reducer->attrs.attr_store.at("dim"));
  if (reducer_axes.empty()) {
    for (int i = 0; i < reducer_shape.size(); ++i) {
      reducer_axes.emplace_back(i);
    }
  }

  VLOG(2) << "master node : " << master->id()
          << " ,reducer node : " << reducer->id();
  for (int idx = sub_group->nodes.size() - 1; idx >= 0; --idx) {
    auto node = sub_group->nodes[idx];

    if (node == master) {
      continue;
    }
    if (op_pattern_dict[node->op()] == framework::kReduction) {
      continue;
    }
    auto node_data = GetNodeData(node);
    auto node_tensor = tensor_map[node_data->id()];

    VLOG(3) << "Schedule node -> " << node->id()
            << " var : " << node_tensor->name;
    // for x86 schedule.
    if (this->target_ == common::DefaultHostTarget()) {
      LOG(FATAL) << "X86 Not implemented";
    }

    bool dont_compute_inline = group->output_nodes.count(node) ||
                               group->internal_nodes.count(node) ||
                               sub_group->internal_nodes.count(node);
    if (!dont_compute_inline) {
      auto consumers = GetConsumers(node);
      for (auto& consumer : consumers) {
        if (op_pattern_dict[consumer->op()] == framework::kReduction) {
          dont_compute_inline = true;
          break;
        }
      }
    }

    // if is const op, do compute inline.
    if (IsConstOp(node) && !group->output_nodes.count(node)) {
      dont_compute_inline = false;
    }

    // if node is internal node or output, try to copy schedule from fellow node
    if (dont_compute_inline) {
      VLOG(2) << "Reduce Schedule for Elementwise Type";
      // if node is not output node, set buffer.
      if (!group->output_nodes.count(node)) {
        auto node_block = ir_sch.GetBlock(node_tensor->name);
        ir_sch.SetBuffer(node_block, "local", true);
      }
      // node is after reduce
      auto node_shape = this->shape_dict_.at(node_data->id());
      auto node_size = std::accumulate(
          node_shape.begin(), node_shape.end(), 1, std::multiplies<int>());
      if (node_shape == master_shape || node_size == master_size) {
        VLOG(2) << "Do Elementwise Type After Reduce!";
        auto loops = ir_sch.GetLoops(node_tensor->name);
        // flat loop and tensor shape
        if (op_pattern_dict[master->op()] == framework::kElementWise) {
          ir_sch.FlattenLoops(loops, true);
        } else {
          ir_sch.FlattenLoops(loops, false);
        }
        // split loop to assign master loop
        std::vector<int> factors;
        auto mloops = ir_sch.GetLoops(master_tensor->name);
        for (auto& loop : mloops) {
          factors.push_back(loop.As<ir::For>()->extent.as_int32());
        }
        loops = ir_sch.GetLoops(node_tensor->name);
        ir_sch.Split(loops.back(), factors);
        // note do simple compute at
        auto node_block = ir_sch.GetBlock(node_tensor->name);
        ir_sch.SimpleComputeAt(node_block, mloops.back());
        continue;
      }
      // do elementwise flat
      auto loops = ir_sch.GetLoops(node_tensor->name);
      if (op_pattern_dict[node->op()] == framework::kElementWise) {
        ir_sch.FlattenLoops(loops, true);
      } else {
        ir_sch.FlattenLoops(loops, false);
      }
      // node is before reduce.
      if (WithoutLastDimInReduce(reducer_shape, reducer_axes)) {
        VLOG(2) << "Reduce Schedule for WithoutLastDimInReduce";
        // find a shape to do simple compute at.
        auto tmp_reducer = reducer;
        auto tmp_reducer_shape = reducer_shape;
        auto tmp_reducer_size = std::accumulate(reducer_shape.begin(),
                                                reducer_shape.end(),
                                                1,
                                                std::multiplies<int>());
        // node shape.
        auto node_shape = this->shape_dict_.at(node_data->id());
        if (node_shape != tmp_reducer_shape && node_size != reduce_size) {
          // try to find the same shape reduce from visited_nodes
          for (auto rnode : group->master_nodes) {
            if (op_pattern_dict[rnode->op()] != framework::kReduction) {
              continue;
            }
            auto shape = this->shape_dict_.at(
                rnode->inlinks_in_order()[0]->source()->id());
            auto size = std::accumulate(
                shape.begin(), shape.end(), 1, std::multiplies<int>());
            if (shape == node_shape || size == node_size) {
              tmp_reducer = rnode;
              tmp_reducer_size = size;
              tmp_reducer_shape = shape;
              break;
            }
          }
        }
        // do split
        CHECK(node_shape == tmp_reducer_shape || node_size == tmp_reducer_size);

        auto loops = ir_sch.GetLoops(node_tensor->name);
        ir_sch.Split(loops.back(), tmp_reducer_shape);

        auto tmp_reducer_data = GetNodeData(tmp_reducer);
        auto tmp_reducer_tensor = tensor_map[tmp_reducer_data->id()];
        // if used block shuffle reduce
        if (tensor_map.count(tmp_reducer_data->id() + "_1")) {
          ScheduleAssignReduceWithoutLast(
              ir_sch, node_tensor->name, tmp_reducer_shape, reducer_axes);
          auto tmp_reducer_tensor_0 = tensor_map[tmp_reducer_data->id() + "_0"];
          auto tmp_reducer_loops_0 =
              ir_sch.GetLoops(tmp_reducer_tensor_0->name);
          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() < tmp_reducer_loops_0.size()) {
            ir_sch.Split(
                node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(),
                   tmp_reducer_loops_0.size())
              << "node loops and reduce loops must be equal!";
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SimpleComputeAt(node_block, tmp_reducer_loops_0.back());
        } else {
          OrderAssignReduce(ir_sch, node_tensor->name, reducer_axes);

          auto node_block = ir_sch.GetBlock(node_tensor->name);
          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() <
              ir_sch.GetLoops(tmp_reducer_tensor->name).size()) {
            ir_sch.Split(
                node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(),
                   ir_sch.GetLoops(tmp_reducer_tensor->name).size())
              << "node loop size and reduce loop size must be equal!";
          ir_sch.SimpleComputeAt(
              node_block, ir_sch.GetLoops(tmp_reducer_tensor->name).back());
        }
      } else {
        VLOG(2) << "Reduce Schedule for WithLastDimInReduce";
        if (tensor_map.count(reducer_data->id() + "_1")) {
          {
            auto node_loops = ir_sch.GetLoops(node_tensor->name);
            ir_sch.Split(node_loops.back(), reducer_shape);
          }

          ScheduleAssignReduceWithLast(
              ir_sch, node_tensor->name, reducer_shape, reducer_axes);
          auto reducer_1_tensor = tensor_map[reducer_data->id() + "_1"];
          auto reducer_1_block = ir_sch.GetBlock(reducer_1_tensor->name);
          auto reducer_1_loops = ir_sch.GetLoops(reducer_1_block);

          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (ir_sch.GetLoops(node_tensor->name).size() <
              ir_sch.GetLoops(reducer_1_block).size()) {
            ir_sch.Split(
                node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }

          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(),
                   ir_sch.GetLoops(reducer_1_block).size())
              << "node loop size and reduce loop size must be equal!"
              << ir_sch.GetModule().GetExprs().at(0);
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SimpleComputeAt(node_block, reducer_1_loops.back());
        } else {
          auto reducer_0_tensor = tensor_map[reducer_data->id() + "_0"];
          auto reducer_0_block = ir_sch.GetBlock(reducer_0_tensor->name);
          auto reducer_0_loops = ir_sch.GetLoops(reducer_0_block);
          {
            auto node_loops = ir_sch.GetLoops(node_tensor->name);
            std::vector<int> factors;
            for (auto& loop : reducer_0_loops) {
              factors.push_back(loop.As<ir::For>()->extent.as_int32());
            }
            ir_sch.Split(node_loops.back(), factors);
          }

          auto node_loops = ir_sch.GetLoops(node_tensor->name);
          if (node_loops.size() < reducer_0_loops.size()) {
            ir_sch.Split(
                node_tensor->name, 0, {-1, ir::GetLoopExtent(node_loops[0])});
          }
          CHECK_EQ(ir_sch.GetLoops(node_tensor->name).size(),
                   reducer_0_loops.size())
              << "node loop size and reduce loop size must be equal!"
              << ir_sch.GetModule().GetExprs().at(0);
          auto node_block = ir_sch.GetBlock(node_tensor->name);
          ir_sch.SimpleComputeAt(node_block, reducer_0_loops.back());
        }
      }
      continue;
    }

    // others elemenwise internal node use compute-inline
    VLOG(2) << "Do Elementwise ComputeInline!";
    auto loops = ir_sch.GetLoops(node_tensor->name);
    if (op_pattern_dict[node->op()] == framework::kElementWise) {
      ir_sch.FlattenLoops(loops, true);
    } else {
      ir_sch.FlattenLoops(loops, false);
    }
    auto node_block = ir_sch.GetBlock(node_tensor->name);
    ir_sch.ComputeInline(node_block);
  }
}

std::vector<ir::LoweredFunc> OpLowerer::IRLowerNonFusibleOp(
    GroupPtr& group, bool apply_impl_schedule) {
  VLOG(3) << "LowerNonFusibleOp Group : " << group->group_id;
  // get input tensor and output tensor
  CHECK(group->nodes.size() || group->fused_sub_groups.size());
  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  auto node = group->fused_sub_groups.size()
                  ? group->fused_sub_groups[0]->nodes.front()
                  : group->nodes.front();
  VLOG(3) << "GetOpFunc of op " << node->id();
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;

  std::vector<ir::Argument> args;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  for (auto& node_data : GetInputNodeData(node)) {
    CHECK(node_data);
    ir::Tensor tensor;
    if (!tensor_map.count(node_data->id())) {
      tensor = GetTensor(node_data, this->type_dict_, this->shape_dict_);
      // record tensor.
      tensor_map[node_data->id()] = tensor;
      // input name.
      group->input_names.push_back(node_data->id());
      // input type.
      args.emplace_back(tensor->buffer, ir::Argument::IO::kInput);
    } else {
      tensor = tensor_map[node_data->id()];
    }
    inputs.push_back(tensor);
    cinn_inputs.push_back(common::CINNValue(tensor));
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  auto node_datas = GetAllNodeData(node);
  for (auto node_data : node_datas) {
    VLOG(3) << "cinn_inputs.push_back " << node_data->id();
    group->output_names.push_back(node_data->id());
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
    cinn_inputs.push_back(common::CINNValue(node_data->id()));
  }

  auto impl = OpStrategy::SelectImpl(cinn_strategy[node->op()](
      node->attrs, inputs, out_types, out_shapes, target_));
  // if node op is custom_call, apply custom_call compute.
  if (node->op()->name == "custom_call") {
    std::string external_api;
    if (node->attrs.attr_store.count("custom_call")) {
      external_api =
          absl::get<std::string>(node->attrs.attr_store.at("custom_call"));
    } else {
      external_api =
          ExternalApiRegistry::Global()->GetExternalApi(node, target_);
    }
    std::vector<common::CINNValue> compute_args = {
        common::CINNValue(group->GetFuncName()),
        common::CINNValue(external_api)};
    common::CINNValuePack pack =
        impl->fcompute(common::CINNValuePack{compute_args});
    CHECK_EQ(pack.size(), 1UL);
    // reset input names as extern api input args can't be remove duplicate.
    group->input_names.clear();
    for (auto& inode : node->inlinks_in_order()) {
      group->input_names.push_back(inode->source()->as<NodeData>()->id());
    }
    return {pack[0].operator ir::Expr().as_lowered_func_ref()};
  }

  common::CINNValuePack pack =
      impl->fcompute(common::CINNValuePack{cinn_inputs});
  for (int i = 0; i < pack->size() - 1; i++) {
    ir::Expr temp = pack[i];
    // checkout whether the tensor is with buffer.
    if (!temp.as_tensor_ref()->buffer.defined() ||
        this->target_ != common::DefaultNVGPUTarget()) {
      inputs.push_back(temp.as_tensor_ref());
      temp.as_tensor_ref()->WithBuffer();
      args.emplace_back(temp.as_tensor_ref()->buffer,
                        ir::Argument::IO::kOutput);
    }
  }

  poly::StageMap stages = pack.back();
  auto func = lang::LowerVec(group->GetFuncName(),
                             stages,
                             inputs,
                             {},
                             {},
                             nullptr,
                             this->target_,
                             true);

  if (apply_impl_schedule) {
    std::vector<common::CINNValue> schedule_inputs;
    // collect tensor
    for (int idx = 0; idx < pack.size() - 1; ++idx) {
      CHECK(pack[idx].is_tensor());
      schedule_inputs.push_back(common::CINNValue(pack[idx]));
    }
    for (auto& f : func) {
      schedule_inputs.push_back(common::CINNValue(f->body));
    }
    // do ast tree schedule
    common::CINNValuePack expr_pack =
        impl->fschedule(common::CINNValuePack{schedule_inputs});

    ir::Expr func_body = expr_pack[0];
    std::vector<std::string> input_output_nodes(group->input_names);
    input_output_nodes.insert(input_output_nodes.end(),
                              group->output_names.begin(),
                              group->output_names.end());
    VLOG(6) << "func.size() = " << func.size()
            << ", expr_pack.size() = " << expr_pack.size();
    VLOG(6) << "args.size() = " << args.size()
            << ", input_output_nodes.size() = " << input_output_nodes.size();
    if (args.size() > input_output_nodes.size()) {
      args = lang::GetArgs(func_body, input_output_nodes);
    }
    std::vector<ir::LoweredFunc> res;
    for (int i = 0; i < expr_pack.size(); i++) {
      ir::Expr func_body = expr_pack[0];
#ifdef CINN_WITH_CUDA
      optim::OptimizeExprGPU(&(func_body));
#endif
      auto temp_buffers = lang::GetTempBuffers(inputs, stages, func_body);
      auto function = ir::_LoweredFunc_::Make(
          group->GetFuncName(), args, func_body, temp_buffers);
      res.push_back(function);
    }
    for (auto& i : res) {
      i = optim::Optimize(Expr(i), target_, false).as_lowered_func_ref();
    }
    return res;
  } else {
    for (auto& f : func) {
#ifdef CINN_WITH_CUDA
      optim::OptimizeExprGPU(&(f->body));
#endif
      f = optim::Optimize(Expr(f), target_, false).as_lowered_func_ref();
    }
    return func;
  }
}

// do compute
void OpLowerer::IRSchedule(
    ir::IRSchedule& ir_sch,
    const GroupPtr& group,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  // topological order.
  auto nodes_set = group->NodeSet();
  auto v_consumers = BuildVirtualConsumer(group, this->shape_dict_);
  auto nodes_in_order =
      BFSTopologicalOrderWithPriority(group, v_consumers, this->shape_dict_);
  // find reducer.
  std::unordered_set<Node*> nodes_inline;
  auto greducer = FindGlobalReducer(nodes_in_order);
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");

  // do schedule
  for (auto node : nodes_in_order) {
    VLOG(4) << "Try FUSION " << node->op()->name;
    // consumers.
    auto consumers = GetConsumersInSet(node, nodes_set);
    const Node* reducer =
        greducer ? FindNearestReducer(node, nodes_set) : greducer;
    if (!reducer && greducer) {
      reducer =
          v_consumers.count(node) ? v_consumers.find(node)->second : reducer;
      if (reducer && op_pattern_dict[reducer->op()] != framework::kReduction) {
        reducer = nullptr;
      }
    }

    auto masters = GetMasters(node, nodes_inline, nodes_set);
    // node can be inline.
    if (CanbeInline(node,
                    consumers,
                    reducer,
                    masters,
                    group,
                    nodes_set,
                    this->shape_dict_)) {
      VLOG(3) << "Before compute inline, ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      auto block = ir_sch.GetBlock(GetNodeData(node)->id());
      ir::ComputeInlineChecker checker(ir_sch, block);
      if (!checker.Check()) {
        checker.BuildDataDependency();
        continue;
      }

      // if exist global reduce node.
      if (greducer) {
        auto loops = ir_sch.GetLoops(GetNodeData(node)->id());
        if (op_pattern_dict[node->op()] == framework::kElementWise) {
          ir_sch.FlattenLoops(loops, true);
        } else {
          ir_sch.FlattenLoops(loops, false);
        }
      }

      ir_sch.ComputeInline(block);
      nodes_inline.insert(node);
      VLOG(3) << "After compute inline, ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      continue;
    }
    // find master to computeat.
    auto master = GetMasterToComputeAt(node,
                                       nodes_in_order,
                                       nodes_inline,
                                       nodes_set,
                                       v_consumers,
                                       this->shape_dict_);
    // assign to reducer/master loop.
    if (reducer) {
      VLOG(3) << "Before assign node " << node->id()
              << " into vertical link reducer " << reducer->id() << ", ir is:\n"
              << ir_sch.GetModule().GetExprs().at(0);
      // if node is vertical with reduce, loop assign reducer.
      LoopAssignReduce(
          ir_sch, node, reducer, this->target_, tensor_map, this->shape_dict_);
    } else if (greducer) {
      auto greducer_out_shape =
          this->shape_dict_.at(greducer->outlinks_in_order()[0]->sink()->id());
      auto node_out_shape =
          this->shape_dict_.at(node->outlinks_in_order()[0]->sink()->id());
      if (std::accumulate(greducer_out_shape.begin(),
                          greducer_out_shape.end(),
                          1,
                          std::multiplies<int>()) !=
          std::accumulate(node_out_shape.begin(),
                          node_out_shape.end(),
                          1,
                          std::multiplies<int>())) {
        LoopAssignReduce(ir_sch,
                         node,
                         greducer,
                         this->target_,
                         tensor_map,
                         this->shape_dict_);
      } else {
        VLOG(3) << "Before assign node " << node->id()
                << " into horizontal link reducer " << greducer->id()
                << ", ir is:\n"
                << ir_sch.GetModule().GetExprs().at(0);
        // if node is horizontal with reduce or node is reduce, loop assign
        // master.
        auto loops = ir_sch.GetLoops(GetNodeData(node)->id());
        if (op_pattern_dict[node->op()] == framework::kElementWise) {
          ir_sch.FlattenLoops(loops, true);
        } else if (op_pattern_dict[node->op()] != framework::kReduction) {
          ir_sch.FlattenLoops(loops, false);
        }

        if (master && op_pattern_dict[node->op()] != framework::kReduction) {
          auto master_loops = ir_sch.GetLoops(GetNodeData(master)->id());
          std::vector<int> splits;
          for (auto loop : master_loops) {
            splits.push_back(loop.As<ir::For>()->extent.as_int32());
          }
          loops = ir_sch.GetLoops(GetNodeData(node)->id());
          ir_sch.Split(loops[0], splits);
        }
      }
    }
    VLOG(3) << "Before loop fusion, ir is:\n"
            << ir_sch.GetModule().GetExprs().at(0);
    // do loop fuse.
    LoopComputeAt(ir_sch,
                  node,
                  master ? master : nodes_in_order.front(),
                  group,
                  this->shape_dict_,
                  tensor_map);
    VLOG(3) << "After loop fusion, ir is:\n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // do vectorize
  auto all_blocks = ir_sch.GetAllBlocks();
  VLOG(4) << "Size of blocks: " << all_blocks.size();
  VLOG(4) << "Op Pattern : " << group->op_pattern_kind;

  // only support first block?
  auto block = all_blocks[0];
  CHECK(block->as<ir::ScheduleBlockRealize>());
  CHECK(block->as<ir::ScheduleBlockRealize>()
            ->schedule_block->as<ir::ScheduleBlock>());
  auto is_tensor_block = true;
  auto tensor_name = block->as<ir::ScheduleBlockRealize>()
                         ->schedule_block->as<ir::ScheduleBlock>()
                         ->name;
  if (!tensor_map.count(tensor_name)) {
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
      // get dtype of vectorized var
      auto dtype = this->type_dict_.at(tensor_name);
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
  SyncThreadWithShared(
      ir_sch, group, nodes_inline, nodes_set, this->shape_dict_, tensor_map);
  VLOG(4) << "After IRSchedule,  ir is: \n"
          << ir_sch.GetModule().GetExprs().at(0);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
