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

#include "paddle/cinn/hlir/framework/op_lowering_impl.h"

#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/op_lowering_util.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);

namespace cinn {
namespace hlir {
namespace framework {

using common::bfloat16;
using common::float16;

using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;
using framework::StrategyFunction;

using common::Type;

using cinn::hlir::op::ExternalApiRegistry;

OpLowererImpl::OpLowererImpl(
    const absl::flat_hash_map<std::string, Type>& type_dict,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict,
    const Target& target)
    : type_dict_(type_dict), shape_dict_(shape_dict), target_(target) {}

std::vector<ir::LoweredFunc> OpLowererImpl::Lower(const GroupPtr& group,
                                                  bool apply_op_schedule,
                                                  bool apply_group_schedule) {
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

bool OpLowererImpl::ElementwiseScheduleDetermineFunction(Node* node) {
  return true;
}

bool OpLowererImpl::ReduceScheduleDetermineFunction(Node* node) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  return op_pattern_dict[node->op()] == framework::kReduction;
}

bool OpLowererImpl::NonFusibleScheduleDetermineFunction(Node* node) {
  return true;
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerGroup(
    const GroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    ScheduleDetermineFunction schedule_determine_func) {
  // 1.Do compute, lower and schedule for each op.
  VLOG(3) << "group->fused_sub_groups.size() is : "
          << group->fused_sub_groups.size();
  std::vector<Node*> nodes = group->CollectNodes();
  if (nodes.size() == 1 && nodes[0]->op()->name == "custom_call") {
    return LowerCustomCall(group);
  }
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  bool do_op_schedule = apply_group_schedule || apply_op_schedule;
  std::vector<ir::Expr> func_bodies = LowerOps(nodes,
                                               do_op_schedule,
                                               schedule_determine_func,
                                               &group_func_arg_tensors,
                                               &tensor_map);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    DoGroupSchedule(ir_sch, group, tensor_map);
    VLOG(3) << "After group schedule, ir is: \n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  return PostProcess(
      group, tensor_map, do_op_schedule, &ir_sch, &group_func_arg_tensors);
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerCustomCall(
    const GroupPtr& group) {
  std::vector<Node*> nodes = group->CollectNodes();
  if (nodes.size() != 1) {
    std::ostringstream err_msg;
    err_msg << "Lowering custom call, group func name: " << group->GetFuncName()
            << ", expect 1 node, but got " << nodes.size();
    std::ostringstream detail_info;
    detail_info << "Node id:";
    for (const Node* node : nodes) {
      detail_info << node->id() << ", ";
    }
    throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                              err_msg.str(),
                              detail_info.str(),
                              __FILE__,
                              __LINE__);
  }
  Node* node = nodes[0];
  std::vector<ir::Tensor> op_func_arg_tensors;
  std::unordered_map<std::string, ir::Tensor> tensor_map;
  for (auto& node_data : GetInputNodeData(node)) {
    if (node_data == nullptr) {
      std::ostringstream err_msg;
      err_msg << "Lowering custom call, group func name: "
              << group->GetFuncName() << ",  one of input node data of "
              << node->id() << " is nullptr";
      throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                                err_msg.str(),
                                "",
                                __FILE__,
                                __LINE__);
    }
    ir::Tensor tensor;
    if (!tensor_map.count(node_data->id())) {
      tensor = GetTensor(node_data, this->type_dict_, this->shape_dict_);
      // record tensor.
      tensor_map[node_data->id()] = tensor;
      // input name.
      group->input_names.push_back(node_data->id());
    } else {
      tensor = tensor_map[node_data->id()];
    }
    op_func_arg_tensors.push_back(tensor);
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  auto node_datas = GetAllNodeData(node);
  for (auto node_data : node_datas) {
    group->output_names.push_back(node_data->id());
    out_types.push_back(this->type_dict_.at(node_data->id()));
    out_shapes.push_back(this->shape_dict_.at(node_data->id()));
  }
  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto impl = OpStrategy::SelectImpl(cinn_strategy[node->op()](
      node->attrs, op_func_arg_tensors, out_types, out_shapes, target_));
  std::string external_api;
  if (node->attrs.attr_store.count("custom_call")) {
    external_api =
        absl::get<std::string>(node->attrs.attr_store.at("custom_call"));
  } else {
    external_api = ExternalApiRegistry::Global()->GetExternalApi(node, target_);
  }
  std::vector<common::CINNValue> compute_args = {
      common::CINNValue(group->GetFuncName()), common::CINNValue(external_api)};
  common::CINNValuePack pack =
      impl->fcompute(common::CINNValuePack{compute_args});
  if (pack.size() != 1) {
    std::ostringstream err_msg;
    err_msg << "Lowering custom call, group func name: " << group->GetFuncName()
            << ", expect 1 pack after executing fcompute, but got "
            << pack.size();
    throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                              err_msg.str(),
                              "",
                              __FILE__,
                              __LINE__);
  }
  // reset input names as extern api input args can't be remove duplicate.
  group->input_names.clear();
  for (auto& inode : node->inlinks_in_order()) {
    group->input_names.push_back(inode->source()->as<NodeData>()->id());
  }
  return {pack[0].operator ir::Expr().as_lowered_func_ref()};
}

std::vector<ir::LoweredFunc> OpLowererImpl::PostProcess(
    const GroupPtr& group,
    const std::unordered_map<std::string, ir::Tensor>& tensor_map,
    bool done_op_schedule,
    ir::IRSchedule* ir_sch,
    std::vector<ir::Tensor>* group_func_arg_tensors) {
  // 1.Prepare function args
  group->input_names.clear();
  std::vector<ir::Argument> group_func_args;
  std::unordered_set<std::string> arg_name_set;
  for (auto& arg_tensor : *group_func_arg_tensors) {
    // input node data name.
    group->input_names.push_back(arg_tensor->name);
    // input args
    group_func_args.emplace_back(arg_tensor->buffer, ir::Argument::IO::kInput);
    arg_name_set.insert(arg_tensor->buffer->name);
  }

  group->output_names.clear();
  for (auto& node : group->output_nodes) {
    // collect all output tensor.
    for (auto node_data : GetAllNodeData(node)) {
      std::string output_node_data_name = node_data->id();
      group->output_names.push_back(output_node_data_name);
      if (tensor_map.count(output_node_data_name) == 0) {
        continue;
      }
      auto tensor = tensor_map.at(output_node_data_name);
      if (arg_name_set.count(tensor->buffer->name) != 0) {
        continue;
      }
      // output arg tensors
      group_func_arg_tensors->push_back(tensor);
      // output args
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
  auto func = ir::_LoweredFunc_::Make(group->GetFuncName(),
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
    const std::vector<Node*>& nodes,
    bool apply_op_schedule,
    ScheduleDetermineFunction schedule_determine_func,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<std::string, ir::Tensor>* tensor_map) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<Expr> func_bodies;
  for (Node* node : nodes) {
    // 1.Select Op impl
    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;
    std::vector<NodeData*> node_datas = GetAllNodeData(node);
    for (const auto& node_data : node_datas) {
      out_types.push_back(this->type_dict_.at(node_data->id()));
      out_shapes.push_back(this->shape_dict_.at(node_data->id()));
    }
    std::vector<ir::Tensor> op_func_arg_tensors =
        std::move(CollectInputTensor(node,
                                     this->type_dict_,
                                     this->shape_dict_,
                                     group_func_arg_tensors,
                                     tensor_map));
    auto op_impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs,
                                                    op_func_arg_tensors,
                                                    out_types,
                                                    out_shapes,
                                                    this->target_));

    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs =
        DoOpLower(op_impl, node, tensor_map, &op_func_arg_tensors);

    if (apply_op_schedule && (this->*schedule_determine_func)(node)) {
      // 3.Perform the schedule of Op
      func_bodies.push_back(DoOpSchedule(op_impl, op_func_arg_tensors, funcs));
    } else {
      for (const ir::LoweredFunc& func : funcs) {
        func_bodies.push_back(func->body);
      }
    }
  }

  return func_bodies;
}

std::vector<ir::LoweredFunc> OpLowererImpl::DoOpLower(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    Node* node,
    std::unordered_map<std::string, ir::Tensor>* tensor_map,
    std::vector<ir::Tensor>* op_func_arg_tensors) {
  VLOG(4) << "Do lower with Compute, op: " << node->op()->name;
  std::vector<common::CINNValue> cinn_inputs;
  for (const ir::Tensor& tensor : *op_func_arg_tensors) {
    cinn_inputs.push_back(common::CINNValue(ir::Expr(tensor)));
  }
  // set tensor name = node data name
  std::vector<NodeData*> node_datas = GetAllNodeData(node);
  for (const NodeData* node_data : node_datas) {
    cinn_inputs.push_back(common::CINNValue(node_data->id()));
  }

  // 1.Do compute
  common::CINNValuePack pack =
      op_impl->fcompute(common::CINNValuePack{cinn_inputs});

  poly::StageMap tmp_stages = pack.back();
  std::string post = "";
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    Expr expr = pack[idx];
    // Insert the output tensor defined by Compute into the tensor_map
    if (pack.size() - 1 > node_datas.size()) {
      // Some nodes may output multiple temp tensors in their Compute
      // definition, but only one output node_data in the graph, and we use id +
      // "_0"/"_1" as key.
      (*tensor_map)[node_datas[0]->id() + post] = expr.as_tensor_ref();
      post = "_" + std::to_string(idx);
    } else {
      // If the number of output tensors defined by Compute is less equal than
      // the output node_data on the graph, then there is a one-to-one
      // correspondence, and the redundant output node_data contact empty.
      (*tensor_map)[node_datas[idx]->id()] = expr.as_tensor_ref();
    }

    // Insert output tensors into function arg
    if (!expr.as_tensor_ref()->buffer.defined() ||
        this->target_ != common::DefaultNVGPUTarget()) {
      op_func_arg_tensors->push_back(expr.as_tensor_ref());
      expr.as_tensor_ref()->WithBuffer();
    }
  }

  // 2.Do lower
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec("fn_" + node->id(),
                                                      tmp_stages,
                                                      *op_func_arg_tensors,
                                                      {},
                                                      {},
                                                      nullptr,
                                                      this->target_,
                                                      true);
  VLOG(4) << "Lower op: " << node->op()->name << ", get " << funcs.size()
          << " LoweredFunc:\n";

  op_func_arg_tensors->clear();
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    if (!pack[idx].is_tensor()) {
      std::ostringstream err_msg;
      err_msg << "Lowering op: " << node->op()->name
              << ", after executing fcompute, pack [" << idx
              << "] is not a tensor.";
      throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                                err_msg.str(),
                                "",
                                __FILE__,
                                __LINE__);
    }
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

// group schedule
ir::Expr OpLowererImpl::DoGroupSchedule(
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
        //
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
  return ir_sch.GetModule().GetExprs().at(0);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
