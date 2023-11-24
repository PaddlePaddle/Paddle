// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/lower_impl.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace lang {
namespace detail {

void CheckNoIslCallRemains(Expr* expr) {
  auto isl_calls = ir::ir_utils::CollectIRNodes(*expr, [](const Expr* expr) {
    return expr->As<ir::Call>() && expr->As<ir::Call>()->is_isl_call();
  });
#ifdef CINN_DEBUG
  for (auto& item : isl_calls) {
    LOG(ERROR) << "ISL call: " << item;
  }
#endif
  if (!isl_calls.empty()) {
    LOG(WARNING) << "Some ISL call nodes remained, get " << isl_calls.size()
                 << " isl_calls, the first one is " << *isl_calls.begin();
  }
}

void BindBuffer(StageMap& stages) {  // NOLINT
  absl::flat_hash_map<std::string, ir::_Tensor_*> tensor_map;
  for (auto& stage : stages) {
    tensor_map[stage.second->tensor()->name] = stage.second->tensor();
  }
  for (auto& stage : stages) {
    if (!stage.second->tensor()->buffer.defined() &&
        !stage.second->meta.tensors_to_share_buffer_with.empty()) {
      for (auto& str : stage.second->meta.tensors_to_share_buffer_with) {
        if (tensor_map[str]->buffer.defined()) {
          auto edited_shape = tensor_map[str]->buffer->shape;
          stage.second->tensor()->Bind(tensor_map[str]->buffer);
          tensor_map[str]->buffer->shape = edited_shape;
          VLOG(3) << "Tensor " << stage.second->tensor()->name
                  << " bind buffer to " << tensor_map[str]->name << " , "
                  << tensor_map[str]->buffer->name;
        }
      }
    }
  }
}

Expr LowerGroup(const poly::ScheduleGroup& group,
                const std::map<std::string, Expr>& tuple_to_expr,
                std::map<std::string, ir::Tensor>* global_tensor_map,
                std::unordered_map<std::string, std::vector<Expr>>&
                    resized_buffer_cache,  // NOLINT
                StageMap stage_map,
                ir::CudaAxisInfo* cuda_axis_info) {
  BindBuffer(stage_map);
  std::vector<poly::Stage*> stages;
  for (auto& node : group.nodes) {
    VLOG(1) << "In LowerGroup, node id is: " << node->id();
    if (node->stage->has_expression()) {
      stages.push_back(node->stage);
      VLOG(1) << "stage expr " << node->stage->expr();
    } else {
      VLOG(1) << "stage expression is null: " << node->stage->domain();
    }
  }

  if (stages.empty()) return Expr();

  // get isl generated expression
  isl::set context(Context::isl_ctx(), "{:}");
  poly::AstGen gen(context, stages, group);
  isl::ast_node ast = gen.Build();
  ir::Expr e;

  // The code where adds length 1 loop back to CINN Expr, if you do not want to
  // add back, call poly::IslAstNodeToCinnExpr(ast, &e) instead of
  // poly::IslAstNodeToCinnExpr(ast, gen.domain(), &e);

  VLOG(6) << "before ast to expr";
  // poly::IslAstNodeToCinnExpr(ast, &e);
  poly::IslAstNodeToCinnExpr(ast, gen.domain(), &e);
  // now we get a workable expression, but the statement are something like
  // `B(((16 * po0) + po1), po2)`, we need to transform this to some realworld
  // statement in CINN.

  VLOG(1) << "ast to expr: \n" << e << std::endl;

  // replace isl call to the corresponding CINN statement, we need to replace
  // the axis at the same time.
  for (auto& statement : tuple_to_expr) {
    VLOG(2) << "LowerGroup working on statement: " << statement.first;
    if (!gen.ContainsStatement(statement.first)) continue;
    // the axis_ast_map contains the axis from the original (like `i`) to the
    // transformed (like `i+3`).
    auto axis_expr_map = gen.axis2expr(statement.first);
    for (auto& item : axis_expr_map) {
      VLOG(4) << "statement ast map axis [" << item.first << "] to "
              << "[" << item.second << "]";
    }

    // the original CINN statements.
    Expr statement_candi_expr = tuple_to_expr.at(statement.first);

    VLOG(3) << "replacing " << statement.first << " to "
            << statement_candi_expr;
    optim::ReplaceIslCallWithExpr(
        &e, statement.first, statement_candi_expr, axis_expr_map);
  }
  CheckNoIslCallRemains(&e);

  // Update global_tensor_map
  for (auto& e : stage_map) {
    if (!global_tensor_map->count(e.second->id())) {
      (*global_tensor_map)[e.second->id()] = ir::Tensor(e.second->tensor());
    }
  }

  // mark vectorize.
  {
    std::map<std::string, ir::VectorizeInfo> vectorizes;
    for (auto& node : group.nodes) {
      if (node->stage->vectorize_info().valid()) {
        vectorizes[node->stage->id()] = node->stage->vectorize_info();
      }
    }
    MarkVectorizeMutator mutator(vectorizes);
    mutator(&e);
  }

  // mark unroll.
  {
    std::map<std::string, std::set<int>> unrolls;
    for (auto& node : group.nodes) {
      if (!node->stage->unroll_info().empty()) {
        unrolls[node->stage->id()] = node->stage->unroll_info();
      }
    }
    MarkUnrollMutator mutator(unrolls);
    mutator(&e);
  }

  // mark parallel.
  {
    std::map<std::string, std::set<int>> parallels;
    for (auto& node : group.nodes) {
      if (!node->stage->parallel_info().empty()) {
        parallels[node->stage->id()] = node->stage->parallel_info();
      }
    }
    MarkParallelMutator mutator(parallels);
    mutator(&e);
  }

  return e;
}

bool TensorContainsGPUInfo(ir::Tensor t, poly::Stage* stage) {
  if (stage->inlined()) return false;
  if (stage) {
    for (auto& info : stage->forloop_infos()) {
      if (info.second.device == ir::DeviceAPI::GPU) {
        return true;
      }
    }
  }
  return false;
}

const char* CompuGraphNode::__type_info__ = "ComputeGraphNode";
const char* CompuGraphNode::type_info() const { return __type_info__; }
std::string CompuGraphNode::id() const {
  CHECK(tensor.defined());
  return tensor->name;
}

/**
 * \brief Add nodes to graph with dependencies.
 * We create a computation graph based on the tensor dependency relations.
 * NOTE The graph will contain the inline tensors so that the dependency will be
 * reserved.
 * @param graph The graph
 * @param t The tensor.
 * @param stages The stage map.
 */
void CreateCompGraphWithInlineTensors(common::Graph* graph,
                                      const ir::Tensor& t,
                                      StageMap stages,
                                      std::set<ir::Tensor>* visited) {
  if (visited->count(t)) return;
  common::GraphNode* t_node = graph->RetrieveNode(t->name);
  if (!t_node) {
    t_node = graph->RegisterNode(t->name, new CompuGraphNode(t));
  }

  visited->insert(t);

  // collect dependency tensors of t
  // here we just collect the tensors in Load nodes
  // NOTE there may be some other cases.
  auto deps = ir::ir_utils::CollectLoadTensors(
      t->body(), [](const Expr* x) { return x->as_tensor(); });
  for (const auto& dep : deps) {
    auto e_tensor = dep.as_tensor_ref();
    auto* e_node = graph->RetrieveNode(e_tensor->name);
    if (!e_node) {
      e_node =
          graph->RegisterNode(e_tensor->name, new CompuGraphNode(e_tensor));
    }
    e_node->Controls(t_node);
    if (!visited->count(e_tensor)) {
      CreateCompGraphWithInlineTensors(graph, e_tensor, stages, visited);
    }
  }
}

std::unique_ptr<common::Graph> CreateCompGraphWithInlineTensorHidden(
    const std::vector<ir::Tensor>& tensors, StageMap stages) {
  // create a graph with inline tensor first.
  std::unique_ptr<common::Graph> graph(new common::Graph);
  std::set<ir::Tensor> visited;
  for (auto& t : tensors) {
    CreateCompGraphWithInlineTensors(graph.get(), t, stages, &visited);
  }

  // greedy remove the inline tensor, each time merge the inputs of an inline
  // tensor to its sink node.

  std::set<common::GraphNode*> inline_nodes;
  do {
    inline_nodes = graph->CollectNodes([&](const common::GraphNode* x) {
      auto* comp_node = x->safe_as<CompuGraphNode>();
      return stages[comp_node->tensor]->inlined();
    });
    if (inline_nodes.empty()) break;

    /*
     * A -> inlined -> B
     * C /
     * =>
     * A -> B
     * C /
     */
    for (auto* inline_node : inline_nodes) {
      // remove this node, merge its inputs to the sink nodes.
      auto inline_inlinks = inline_node->inlinks();
      auto inline_outlinks = inline_node->outlinks();

      // unlink the inline node from its inputs and outputs
      for (auto& link : inline_inlinks) {
        link->source()->UnLinkSingleTo(link->sink());
      }
      for (auto& link : inline_outlinks) {
        link->source()->UnLinkSingleTo(link->sink());
      }

      // link inline node's input nodes to its output nodes.
      for (auto out_edge : inline_outlinks) {
        auto* out = out_edge->sink();
        for (auto in_edge : inline_inlinks) {
          auto* source = in_edge->source();
          source->LinkTo(out);
        }
      }

      graph->DropNode(inline_node);
    }
  } while (!inline_nodes.empty());

  return graph;
}

void CompuGraphAddCtrlDepLinks(common::Graph* graph, StageMap stages) {
  for (auto& x : graph->nodes()) {
    auto* node = x->safe_as<CompuGraphNode>();
    CHECK(node);
    for (auto& dep : stages[node->tensor]->ctrl_depends()) {
      auto* dep_node = graph->RetrieveNode(dep->name);
      if (dep_node) {
        VLOG(3) << "Add control link: " << dep << " -> " << node->id();
        dep_node->Controls(node);
      }
    }
  }
}

std::unique_ptr<common::Graph> CreateCompGraph(
    const std::vector<ir::Tensor>& tensors, StageMap stages, bool hide_inline) {
  if (hide_inline) {
    auto graph = CreateCompGraphWithInlineTensorHidden(tensors, stages);
    CompuGraphAddCtrlDepLinks(graph.get(), stages);
    return graph;
  } else {
    auto graph = std::make_unique<common::Graph>();
    std::set<ir::Tensor> visited;
    for (auto& t : tensors) {
      CreateCompGraphWithInlineTensors(graph.get(), t, stages, &visited);
    }
    CompuGraphAddCtrlDepLinks(graph.get(), stages);
    return graph;
  }
}

void LowerImpl::CheckArgsUnique() {
  for (auto& tensor : tensor_args_) {
    CHECK(!stages_[tensor]->inlined())
        << "Inline tensor cannot be argument of function";
    if (!tensor->buffer.defined()) {
      LOG(ERROR) << "tensor [" << tensor->name << "] buffer is null";
      continue;
    }
  }
}

std::vector<ir::Argument> LowerImpl::GenerateFunctionArgumentList(
    Expr fn_body) {
  CheckArgsUnique();

  std::vector<ir::Argument> args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&fn_body);

  std::set<std::string> arg_names;

  for (auto& scalar : scalar_args_) {
    CHECK(!arg_names.count(scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    CHECK(scalar_node->type().valid());
    arg_names.insert(scalar->name);

    args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  for (auto& tensor : tensor_args_) {
    auto* tensor_node = tensor.As<ir::_Tensor_>();
    bool is_output = teller.count(tensor->name);
    VLOG(1) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor_node->buffer.defined()) continue;
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor_node->buffer->name)) {
      auto it =
          std::find_if(args.begin(), args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor_node->buffer->name;
          });
      CHECK(it != args.end());
      if (it->is_input()) {
        args.erase(it);
      } else if (it->is_output()) {
        continue;
      }
    }

    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(3) << "Collect " << (is_output ? "W" : "R") << " argument "
            << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }

  return args;
}
// Generate Function Arguments for splitted kernel.
std::vector<ir::Argument> LowerImpl::GenFuncArgForSplitKernel(
    Expr func_iterator, std::vector<ir::Tensor> temp_tensors) {
  CheckArgsUnique();

  std::vector<ir::Argument> in_args;
  std::vector<ir::Argument> out_args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&func_iterator);
  std::set<std::string> arg_names;
  std::set<std::string> all_tensor_names;

  for (auto& scalar : scalar_args_) {
    CHECK(!arg_names.count(scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    CHECK(scalar_node->type().valid());
    arg_names.insert(scalar->name);

    in_args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  auto all_tensors =
      ir::ir_utils::CollectIRNodes(func_iterator, [&](const Expr* x) {
        return x->as_tensor() && !stages_[x->as_tensor()]->inlined();
      });

  auto all_vars = ir::ir_utils::CollectIRNodes(
      func_iterator, [&](const Expr* x) { return x->as_var(); });

  for (auto& i : all_tensors) {
    auto* tensor = i.as_tensor();
    all_tensor_names.insert(tensor->name);
    VLOG(3) << "In all_tensors, it has : " << tensor->name;
    if (!stages_[tensor]->meta.tensors_to_share_buffer_with.empty()) {
      for (auto& i : stages_[tensor]->meta.tensors_to_share_buffer_with) {
        all_tensor_names.insert(i);
        VLOG(3) << "And its share_buffer_tensor is : " << i;
      }
    }
  }
  for (auto& i : all_vars) {
    auto* var = i.as_var();
    VLOG(3) << "In all_vars, it has : " << var->name;
  }

  for (auto& i : scalar_args_) {
    VLOG(3) << "In scalar_args_, var has : " << i->name;
  }

  std::set<std::string> temp_tensor_names;

  for (auto& i : temp_tensors) {
    VLOG(3) << "In temp_tensors, it has : " << i->name;
    temp_tensor_names.insert(i->name);
  }

  for (auto& tensor : tensor_args_) {
    VLOG(3) << "In tensor_args_, it has : " << tensor->name;
    if (temp_tensor_names.count(tensor->name) > 0) continue;
    if (all_tensor_names.count(tensor->name) == 0) continue;
    bool is_output = teller.count(tensor->name);
    VLOG(3) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor->buffer.defined()) {
      VLOG(3) << "tensor->buffer is not defined";
      continue;
    }
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor->buffer->name)) {
      auto it = std::find_if(
          in_args.begin(), in_args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor->buffer->name;
          });
      if (it != in_args.end()) {
        in_args.erase(it);
      } else {
        continue;
      }
    }

    arg_names.insert(tensor->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    if (io == ir::Argument::IO::kInput)
      in_args.emplace_back(tensor->buffer, io);
    else
      out_args.emplace_back(tensor->buffer, io);
  }
  if (out_args.empty()) {
    for (auto& i : all_tensors) {
      auto* tensor = i.as_tensor();
      VLOG(3) << "Tensor " << tensor->name;
      if (tensor->buffer.defined() && !arg_names.count(tensor->buffer->name)) {
        bool is_output =
            teller.count(tensor->name) && teller.count(tensor->name);
        if (is_output)
          out_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      }
    }
  }

  std::vector<ir::Argument> args(in_args.begin(), in_args.end());
  args.insert(std::end(args), out_args.begin(), out_args.end());
  return args;
}

std::vector<Tensor> LowerImpl::CollectTemporaryTensors() {
  // a temporary should be in the comp_graph but not contained in the
  // tensor_args.
  absl::flat_hash_map<std::string, Tensor> tensor_arg_map = GenTensorArgMap();
  absl::flat_hash_map<std::string, Tensor> temp_tensor_map;

  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    if (!tensor_arg_map.count(cnode->tensor->name)) {
      temp_tensor_map[cnode->tensor->name] = cnode->tensor;
    }
  }

  std::vector<Tensor> temp_tensors;
  std::transform(
      temp_tensor_map.begin(),
      temp_tensor_map.end(),
      std::back_inserter(temp_tensors),
      [&](const decltype(temp_tensor_map)::value_type& x) { return x.second; });
  return temp_tensors;
}

absl::flat_hash_map<std::string, Tensor> LowerImpl::GenTensorArgMap() {
  absl::flat_hash_map<std::string, Tensor> map;
  for (auto& t : tensor_args_) {
    map[t->name] = t;
  }
  return map;
}

absl::flat_hash_map<std::string, Tensor> LowerImpl::GenAllTensorMap() {
  absl::flat_hash_map<std::string, Tensor> map;
  for (auto& t : CollectAllTensors()) {
    map[t->name] = t;
  }
  return map;
}

std::vector<ir::LoweredFunc> LowerImpl::operator()() {
  std::vector<poly::Stage*> stages;
  std::map<std::string, ir::Tensor> all_tensor_map;
  for (auto& t : CollectAllTensors()) {
    all_tensor_map[t->name] = t;
    if (!stages_[t]->inlined()) stages.push_back(stages_[t]);
  }

  auto deps = CollectExtraDependencies();
  auto schedule =
      poly::CreateSchedule(stages,
                           poly::ScheduleKind::Poly,
                           std::vector<std::pair<std::string, std::string>>(
                               deps.begin(), deps.end()));
  auto func_body = GenerateFunctionBody(schedule.get());

  std::vector<ir::LoweredFunc> result;
  int num_func = 0;
  for (auto& func_iterator : func_body) {
    if (support_ir_schedule_) {
      // add ScheduleBlockRealize
      func_iterator = ir::ScheduleBlockRealize::Make(
          {},
          ir::ScheduleBlock::Make(
              {}, {}, {}, common::UniqName("root"), func_iterator));
    }
    std::set<std::string> temp_tensor_names;
    for (auto& t : temp_tensor_args_) temp_tensor_names.insert(t->name);

    auto tensor_map = optim::InitialAssignBuffer(&func_iterator,
                                                 stages_,
                                                 all_tensor_map,
                                                 comp_graph(),
                                                 temp_tensor_names);
    // copy the tensor(with buffer assigned) back to func's args.
    {
      for (auto& arg : tensor_args_) {
        if (arg->is_placeholder_node()) continue;
        if (arg->buffer.defined()) continue;
        if (arg->body().As<ir::Call>() && arg->body().type().is_void())
          continue;  // extern call
        if (tensor_map.find(arg->name) == tensor_map.end()) {
          LOG(INFO) << "Didn't find arg tensor " << arg->name
                    << "in tensor_map.\n"
                    << "The function is " << fn_name_
                    << "\nAnd all the arg tensors are:\n";
          for (auto& i : tensor_args_) {
            LOG(INFO) << i->name;
          }
          LOG(FATAL) << "Fatal Error!";
        }
        Reference(&arg)->buffer = tensor_map.at(arg->name)->buffer;
      }
    }
    auto store_exprs = ir::ir_utils::CollectIRNodes(
        func_iterator, [](const Expr* x) { return x->As<ir::Store>(); });
    std::vector<ir::Tensor> new_temp_tensors;
    for (auto& expr : store_exprs) {
      auto* store_node = expr.As<ir::Store>();
      CHECK(store_node);
      auto* tensor = store_node->tensor.As<ir::_Tensor_>();
      CHECK(tensor);
      VLOG(3) << "In store_exprs, its name is : " << tensor->name;
      CHECK(tensor->buffer.defined());
      if (tensor->buffer->memory_type != ir::MemoryType::Heap) {
        new_temp_tensors.push_back(store_node->tensor.as_tensor_ref());
      }
    }

    auto func_temp_tensors = CollectTemporaryTensors();
    std::vector<ir::Buffer> temp_buffers;
    std::unordered_set<std::string> buffer_name_set;
    // TODO(Superjomn) write buffer latter.

    if (target_ == common::DefaultNVGPUTarget()) {
      for (auto& t : new_temp_tensors) {
        if (!tensor_map.count(t->name)) continue;
        auto& tt = tensor_map.at(t->name);
        if (tt->buffer.defined() && !buffer_name_set.count(tt->buffer->name)) {
          temp_buffers.push_back(tt->buffer);
          buffer_name_set.insert(tt->buffer->name);
        }
      }
    } else {
      for (auto& t : func_temp_tensors) {
        if (!tensor_map.count(t->name)) continue;
        auto& tt = tensor_map.at(t->name);
        if (tt->buffer.defined() && !buffer_name_set.count(tt->buffer->name)) {
          temp_buffers.push_back(tt->buffer);
          buffer_name_set.insert(tt->buffer->name);
        }
      }
    }

    ir::LoweredFunc func;
    if (target_ == common::DefaultNVGPUTarget()) {
      auto func_args2 =
          GenFuncArgForSplitKernel(func_iterator, new_temp_tensors);
      std::string new_fn_name = fn_name_;
      if (num_func > 0) {
        new_fn_name += "_" + std::to_string(num_func);
      }
      VLOG(3) << "Making func :" << new_fn_name;
      for (auto& i : func_args2) {
        VLOG(3) << "func_args2 is : " << i.name();
      }
      for (auto& i : temp_buffers) {
        VLOG(3) << "temp_buffers is : " << i->name;
      }
      func = ir::_LoweredFunc_::Make(
          new_fn_name, func_args2, func_iterator, temp_buffers);
    } else {
      auto func_args = GenerateFunctionArgumentList(func_iterator);
      func = ir::_LoweredFunc_::Make(
          fn_name_, func_args, func_iterator, temp_buffers);
    }

    if (support_ir_schedule_) {
      optim::TransformPolyForToFor(&func->body);
      optim::SimplifyBlocks(&func->body);
      func->body = ir::Block::Make({func->body});
      result.push_back(ir::LoweredFunc(func.get()));
      num_func++;
    } else {
      optim::ComputeInlineExpand(&func->body, stages_, &all_tensor_map);
      auto res = optim::Optimize(func,
                                 target_,
                                 FLAGS_cinn_runtime_display_debug_info,
                                 /* remove_gpu_for_loops = */ false);

      if (cuda_axis_info_.size() > num_func &&
          cuda_axis_info_[num_func].valid()) {
        auto* res_func = res.as_lowered_func();
        res_func->cuda_axis_info = cuda_axis_info_[num_func];
      }
      result.push_back(ir::LoweredFunc(res.get()));
      num_func++;
    }
  }
  return result;
}

std::vector<Tensor> LowerImpl::CollectAllTensors() {
  std::vector<Tensor> tensors;
  auto topo_order = compu_graph_->topological_order();  // NOLINT
  auto& nodes = std::get<0>(topo_order);
  auto& edges = std::get<1>(topo_order);
  for (auto* node : nodes) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    tensors.push_back(cnode->tensor);
  }
  return tensors;
}

std::set<std::pair<std::string, std::string>>
LowerImpl::CollectExtraDependencies() const {
  std::set<std::pair<std::string, std::string>> deps;
  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    CHECK(cnode);
    for (auto& dep : stages_[cnode->tensor]->ctrl_depends()) {
      deps.emplace(dep->name, cnode->tensor->name);
    }
  }
  return deps;
}

std::vector<Expr> LowerImpl::GenerateFunctionBody(
    const poly::Schedule* schedule) {
  // generate the expressions for each group.
  std::vector<Expr> exprs;
  std::vector<Expr> result;
  auto tensor_map = GenAllTensorMap();
  std::map<std::string, Expr> tuple_to_expr;
  CHECK(!schedule->groups.empty()) << "no group is generated";

  std::map<std::string, ir::Tensor> global_tensor_map;
  std::unordered_map<std::string, std::vector<Expr>> resized_buffer_cache;

  for (auto& group : schedule->groups) {
    CHECK_GT(group.nodes.size(), 0) << "group is empty";
    bool all_temp_tensor = true;
    for (auto& node : group.nodes) {
      if (!tensor_map.count(node->id())) {
        VLOG(2) << "tensor_map doesn't count " << node->id();
        continue;
      }
      auto& tensor = tensor_map[node->id()];
      if (!tensor->has_expression()) continue;
      all_temp_tensor =
          all_temp_tensor &&
          (stages_[tensor]->inlined() ||
           (tensor->buffer.defined() &&
            (tensor->buffer->memory_type == ir::MemoryType::GPUShared ||
             tensor->buffer->memory_type == ir::MemoryType::GPULocal)));
      auto store_body = tensor->tensor_store_expanded_body();
      if (support_ir_schedule_) {
        // add schedule block of tensor computation for schedule IR
        int var_counts = tensor->shape.size() + tensor->reduce_axis.size();
        std::vector<int> int_shape;
        VLOG(3) << "Tensor " << tensor->name
                << "'s shape is : " << utils::Join(tensor->shape, ",");
        for (auto& expr : tensor->shape) {
          CHECK(expr.is_constant());
          int_shape.push_back(static_cast<int>(expr.get_constant()));
        }
        for (auto& var : tensor->reduce_axis) {
          CHECK(var->lower_bound.defined());
          CHECK(var->upper_bound.defined());
          CHECK(common::is_zero(var->lower_bound));
          CHECK(var->upper_bound.is_constant());
          int_shape.push_back(
              static_cast<int>(var->upper_bound.get_constant()));
        }
        // create block itervars, i0,i1...
        std::vector<Var> block_vars;
        std::vector<Expr> iter_values;
        std::vector<Var> axis_vars =
            common::GenDefaultAxis(tensor->shape.size());
        // bind var_values
        axis_vars.insert(axis_vars.end(),
                         tensor->reduce_axis.begin(),
                         tensor->reduce_axis.end());
        for (int i = 0; i < var_counts; i++) {
          block_vars.push_back(Var(Expr(0),
                                   Expr(int_shape[i]),
                                   cinn::UniqName("i" + std::to_string(i)),
                                   false));
          if (i >= tensor->shape.size()) {
            block_vars[i]->is_reduce_axis = true;
            axis_vars[i]->is_reduce_axis = true;
          }
          iter_values.push_back(axis_vars[i]);
          // replace store's indice
          VLOG(3) << "replace axis_var " << axis_vars[i]->name
                  << " to block_var " << block_vars[i];
          optim::ReplaceVarWithExpr(&store_body, axis_vars[i], block_vars[i]);
        }
        store_body = ir::ScheduleBlockRealize::Make(
            iter_values,
            ir::ScheduleBlock::Make(
                block_vars, {}, {}, tensor->name, store_body));
        // iter_values, ir::ScheduleBlock::Make(block_vars, {}, {},
        // common::UniqName(tensor->name), store_body));
        VLOG(3) << "store body\n" << store_body;
      }
      tuple_to_expr[tensor->name] = store_body;
    }

    ir::CudaAxisInfo temp_cuda_axis_info;
    Expr group_expr = LowerGroup(group,
                                 tuple_to_expr,
                                 &global_tensor_map,
                                 resized_buffer_cache,
                                 stages_,
                                 &temp_cuda_axis_info);

    if (group_expr.defined()) {
      cuda_axis_info_.emplace_back(std::move(temp_cuda_axis_info));
      if (target_ == common::DefaultNVGPUTarget() && !all_temp_tensor) {
        exprs.push_back(group_expr);
        Expr body = ir::Block::Make(exprs);
        result.push_back(body);
        exprs.clear();
      } else {
        exprs.push_back(group_expr);
      }
    }
  }
  if (target_ == common::DefaultHostTarget()) {
    Expr body = ir::Block::Make(exprs);
    result.push_back(body);
    exprs.clear();
  } else if (!exprs.empty()) {
    Expr body = ir::Block::Make(exprs);
    result.push_back(body);
    exprs.clear();
  }

  return result;
}

LowerImpl::LowerImpl(const std::string& fn_name,
                     StageMap stages,
                     const std::vector<Tensor>& tensor_args,
                     const std::vector<Var>& scalar_args,
                     const std::vector<Tensor>& temp_tensor_args,
                     const Target& target,
                     bool support_ir_schedule)
    : fn_name_(fn_name),
      stages_(stages),
      tensor_args_(tensor_args),
      scalar_args_(scalar_args),
      temp_tensor_args_(temp_tensor_args),
      target_(target),
      support_ir_schedule_(support_ir_schedule) {
  // Todo: Here insert auto syncthreads() @haoze

  {  // Update schedule
    std::vector<ir::Tensor> tensors(tensor_args.begin(), tensor_args.end());
    tensors.insert(
        std::end(tensors), temp_tensor_args_.begin(), temp_tensor_args_.end());
    compu_graph_ = CreateCompGraph(tensors, stages, true /*inline_hide*/);

    VLOG(1) << "Computation Graph:\n" << compu_graph_->Visualize();
  }
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn
