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

#include "paddle/cinn/hlir/framework/op_lowering.h"

#include "paddle/cinn/hlir/framework/op_lowering_util.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/ir/ir_schedule.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"

DECLARE_bool(cinn_ir_schedule);
DECLARE_bool(cinn_use_cuda_vectorize);

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
using namespace lang;

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
        return IRLowerOp(&OpLowerer::IRElementwiseCompute, group);
      case framework::kReduction:
        return IRLowerOp(&OpLowerer::IRReduceCompute, group);
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

// group schedule
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
