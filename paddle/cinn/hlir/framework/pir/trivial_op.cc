// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/trivial_op.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

// #include "paddle/cinn/frontend/group_pattern_util.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops) {
  const auto& op_pattern_map =
      Operator::GetAttrs<cinn::hlir::framework::OpPatternKind>("OpPattern");
  std::vector<OpPatternKind> op_patterns;
  const auto ConvertToPattern = [&op_pattern_map](const ::pir::Operation* op) {
    const std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    return op_pattern_map[cinn_op];
  };
  std::transform(ops.begin(),
                 ops.end(),
                 std::back_inserter(op_patterns),
                 ConvertToPattern);
  return op_patterns;
}

template <class A, class C, class Func>
void SequenceMutator(const std::vector<A>& as, C* acc, const Func& mutator) {
  VLOG(4) << "SequenceTransform Init: " << acc;
  for (int i = 0; i < as.size(); ++i) {
    mutator(as[i], acc);
    VLOG(4) << "SequenceTransform Iter: " << acc;
  }
}

static bool IsAdjecent(const ir::Expr& upstream, const ir::Expr& downstream) {
  // 1. Get inputs / output from Expr, then we can tell whether they are
  // adjecent.
  std::set<Expr> upstream_stores =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          upstream, [](const Expr* expr) {
            return expr->As<ir::Store>() &&
                   expr->As<ir::Store>()->is_addr_tensor();
          });
  // don't support multi-output yet.
  PADDLE_ENFORCE(upstream_stores.size() == 1,
                 "The expr of injective should have only one store");

  std::set<Expr> downstream_loads =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          downstream, [](const Expr* expr) {
            return expr->As<ir::Load>() &&
                   expr->As<ir::Load>()->is_addr_tensor();
          });

  for (const auto& upstream_store : upstream_stores) {
    for (const auto& downstream_load : downstream_loads) {
      if (upstream_store.As<ir::Store>()->tensor.As<ir::_Tensor_>()->name ==
          downstream_load.As<ir::Load>()->tensor.As<ir::_Tensor_>()->name) {
        return true;
      }
    }
  }
  return false;
}

inline bool IsTrivialKind(OpPatternKind kind) {
  return kind == OpPatternKind::kElementWise ||
         kind == OpPatternKind::kBroadcast || kind == OpPatternKind::kInjective;
}


void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns) {
  if (VLOG_IS_ON(4)) {
    for (const auto& func : op_compute_bodies) {
      VLOG(4) << "TrivialOpFusion: {FuncBody is} :" << func;
    }
    for (const auto& op_ptn : op_patterns) {
      VLOG(4) << "OpPattern is :" << op_ptn;
    }
  }
  VLOG(4) << "      op_patterns.size() = " << op_compute_bodies.size();
  VLOG(4) << "op_compute_bodies.size() = " << op_patterns.size();
  PADDLE_ENFORCE_EQ(
      op_patterns.size(), op_compute_bodies.size(), "ops and  size not equal");
}

namespace ComposeUtils{

struct MappingLoadStoreExprToDestExprMutator : public ir::IRMutator<> {
  explicit MappingLoadStoreExprToDestExprMutator(const ir::Expr& source,
                                                 const ir::Expr& dest)
      : source_(source), dest_(dest) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* load, Expr* op) override {
    if (load == source_.ptr()) {
      VLOG(4) << "substitude find!";
      *op = dest_;
    } else {
      IRMutator::Visit(load, op);
    }
  }
  void Visit(const ir::Store* store, Expr* op) override {
    if (store == source_.ptr()) {
      VLOG(4) << "substitude find!";
      *op = dest_;
    } else {
      IRMutator::Visit(store, op);
    }
  }

 private:
  ir::Expr source_;
  ir::Expr dest_;
};

static Expr CopyedReplaceExpr(const Expr& source,
                              const std::vector<Var>& replaced,
                              const std::vector<Expr>& candidates) {
  CHECK_EQ(replaced.size(), candidates.size())
      << "In ReplaceExpr, the size of Vars to be replaced must be equal to "
          "the "
          "size of cadidate Exprs! Please check.";
  auto copyed_source = ir::ir_utils::IRCopy(source);
  if (replaced.empty()) return copyed_source;
  std::map<Var, Expr, ir::CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  ir::MappingVarToExprMutator mapper(replacing_map);
  mapper(&copyed_source);
  return copyed_source;
}

static void SubstitudeTargetExprWithDestExpr(const ir::Expr& source,
                                              const ir::Expr& dest,
                                              ir::Expr* body) {
  VLOG(4) << "Start SubstitudeTargetExprWithDestExpr";
  MappingLoadStoreExprToDestExprMutator mapper(source, dest);
  mapper(body);
  VLOG(4) << "End SubstitudeTargetExprWithDestExpr";
}

static ir::Expr SubstitudeIndexVector(const Expr& source,
                                        const std::vector<Var>& load_vars,
                                        const std::vector<ir::Expr>& indices) {
  return CopyedReplaceExpr(source, load_vars, indices);
}

template<typename FusionOp>
static void ReplaceDownstreamLoadExprWithUpstreamComputeBody(
    const FusionOp& upstream,
    const ir::Expr& downstream_load_expr,
    ir::Expr* downstream_body) {
  ComposeUtils::SubstitudeTargetExprWithDestExpr(
      downstream_load_expr,
      ComposeUtils::SubstitudeIndexVector(upstream.GetStoreValue(), 
        upstream.GetOutputIters(), downstream_load_expr.As<ir::Load>()->indices),
      downstream_body);
}

std::set<Expr> GetStoreFromBody(const ir::Expr& body) {
  std::set<Expr> store_tensor_exprs =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          body, [](const Expr* expr) {
            return expr->As<ir::Store>() &&
                    expr->As<ir::Store>()->is_addr_tensor();
          });
  
  return store_tensor_exprs;
}

}

struct TrivialOp {
 public:
  explicit TrivialOp(const ir::Expr& origin_func_body) {
    func_body = ir::ir_utils::IRCopy(origin_func_body);
  }

  ir::Expr GetStoreValue() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  ir::Expr* GetStoreValuePointer() const {
    return &GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  std::vector<ir::Var> GetOutputIters() const {
    std::vector<ir::Var> vars;
    const auto& indices = GetSingleStoreExpr(func_body).As<ir::Store>()->indices;
    std::transform(indices.begin(),
                   indices.end(),
                   std::back_inserter(vars),
                   [](const ir::Expr& expr) { return expr.as_var_ref(); });
    return vars;
  }

  ir::Expr GetFuncBody() const { return func_body; }

  ir::Tensor GetOutputTensor() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->tensor.as_tensor_ref();
  }

  std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Tensor& tensor) const {
    VLOG(4) << "Start GetEachTensorLoadExpr: " << tensor;
    std::set<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
        GetStoreValue(), [&tensor](const Expr* expr) {
          return expr->As<ir::Load>() &&
                 expr->As<ir::Load>()->is_addr_tensor() &&
                 expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     tensor->name;
        });
    for (auto& t : load_exprs) {
      VLOG(4) << "GetEachTensorLoadExpr: " << t << " " << t.ptr();
    }
    return std::vector(load_exprs.begin(), load_exprs.end());
  }

 private:
  ir::Expr func_body;

  ir::Expr GetSingleStoreExpr(const ir::Expr& body) const{
      const auto& store_tensor_exprs = ComposeUtils::GetStoreFromBody(body);
      PADDLE_ENFORCE(store_tensor_exprs.size() == 1,
                  "TrivialOp must store for output only once.");
      return *(store_tensor_exprs.begin());
  }

};

struct ReduceOp {
 public:
  explicit ReduceOp(const ir::Expr& origin_func_body) {
    func_body = ir::ir_utils::IRCopy(origin_func_body);
  }

  ir::Expr GetStoreValue() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  ir::Expr* GetStoreValuePointer() const {
    return &GetSingleStoreExpr(func_body).As<ir::Store>()->value;
  }

  std::vector<ir::Var> GetOutputIters() const {
    std::vector<ir::Var> vars;
    const auto& indices = GetSingleStoreExpr(func_body).As<ir::Store>()->indices;
    std::transform(indices.begin(),
                   indices.end(),
                   std::back_inserter(vars),
                   [](const ir::Expr& expr) { return expr.as_var_ref(); });
    return vars;
  }

  ir::Expr GetFuncBody() const { return func_body; }

  ir::Tensor GetOutputTensor() const {
    return GetSingleStoreExpr(func_body).As<ir::Store>()->tensor.as_tensor_ref();
  }

  std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Tensor& tensor) const {
    VLOG(4) << "Start GetEachTensorLoadExpr: " << tensor;
    std::set<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
        GetStoreValue(), [&tensor](const Expr* expr) {
          return expr->As<ir::Load>() &&
                 expr->As<ir::Load>()->is_addr_tensor() &&
                 expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     tensor->name;
        });
    for (auto& t : load_exprs) {
      VLOG(4) << "GetEachTensorLoadExpr: " << t << " " << t.ptr();
    }
    return std::vector(load_exprs.begin(), load_exprs.end());
  }

 private:
  ir::Expr func_body;

  ir::Expr GetSingleStoreExpr(const ir::Expr& body) const{
    std::vector<ir::Expr> store_tensor_exprs;
    for(const ir::Expr& store_expr: ComposeUtils::GetStoreFromBody(body)){
      std::string store_name = store_expr.As<ir::Store>()->tensor.As<ir::_Tensor_>()->name;
      if (store_name.find("reduce_init") != std::string::npos)
        continue;
      store_tensor_exprs.emplace_back(store_expr);
    }

    PADDLE_ENFORCE(store_tensor_exprs.size() == 1,
                "ReduceOp must store for output only once.");
    return *(store_tensor_exprs.begin());
  }
};

ir::Expr TTFusion(ir::Expr upper, ir::Expr down) {
  VLOG(4) << "TTFusion begin.";
  TrivialOp upstream(upper);
  TrivialOp downstream(down);
  const auto& replaced_tensor = upstream.GetOutputTensor();
  VLOG(4) << "connected tensor is:" << replaced_tensor;
  VLOG(4) << "store value is :" << downstream.GetStoreValue();
  VLOG(4) << "upper :\n" << upper;
  VLOG(4) << "down :\n" << down;

  TrivialOp fused(ir::ir_utils::IRCopy(downstream.GetFuncBody()));
  SequenceMutator(
      fused.GetEachTensorLoadExpr(replaced_tensor),
      fused.GetStoreValuePointer(),
      [&](const ir::Expr& downstream_load_expr, ir::Expr* downstream_body) {
        ComposeUtils::ReplaceDownstreamLoadExprWithUpstreamComputeBody(
            upstream, downstream_load_expr, downstream_body);
      });

  VLOG(4) << "After mutate, store_value is: " << fused.GetFuncBody();
  VLOG(4) << "TTFusion end:\n" << fused.GetFuncBody();
  return fused.GetFuncBody();
}

ir::Expr TRFusion(ir::Expr upper, ir::Expr down) {
  VLOG(4) << "TRFusion begin.";
  TrivialOp upstream(upper);
  ReduceOp downstream(down);
  const auto& replaced_tensor = upstream.GetOutputTensor();
  VLOG(4) << "connected tensor is:" << replaced_tensor;
  VLOG(4) << "store value is :" << downstream.GetStoreValue();

  VLOG(4) << "upper :\n" << upper;
  VLOG(4) << "down :\n" << down;

  ReduceOp fused(ir::ir_utils::IRCopy(downstream.GetFuncBody()));
  SequenceMutator(
      fused.GetEachTensorLoadExpr(replaced_tensor),
      fused.GetStoreValuePointer(),
      [&](const ir::Expr& downstream_load_expr, ir::Expr* downstream_body) {
        ComposeUtils::ReplaceDownstreamLoadExprWithUpstreamComputeBody(
            upstream, downstream_load_expr, downstream_body);
      });

  VLOG(4) << "TRFusion end:\n" << fused.GetFuncBody();
  return fused.GetFuncBody();
}

struct FusionNode {
  // Function bodies losses the kind information which needed in trivialop
  // fusion.
  std::vector<ir::Expr> op_compute_body;
  OpPatternKind op_pattern;

  ::pir::Operation* expr_related_op;

  std::unordered_map<FusionNode*, ::pir::Value> upstream;
  std::unordered_map<FusionNode*, ::pir::Value> downstream;

  explicit FusionNode(ir::Expr op_compute_body, OpPatternKind op_pattern)
      : op_compute_body({op_compute_body}), op_pattern(op_pattern) {}

  void replace_topo_structure_of_fused_nodes(FusionNode* fused_up_node, FusionNode* fused_down_node){
    upstream.insert(fused_up_node->upstream.begin(), fused_up_node->upstream.end());
    upstream.insert(fused_down_node->upstream.begin(), fused_down_node->upstream.end());
    upstream.erase(fused_up_node);

    downstream.insert(fused_up_node->downstream.begin(), fused_up_node->downstream.end());
    downstream.insert(fused_down_node->downstream.begin(), fused_down_node->downstream.end());
    downstream.erase(fused_down_node);

    expr_related_op = fused_down_node->expr_related_op;

    for (const auto& pair_data: upstream){
      FusionNode* upstream_node = pair_data.first;
      ::pir::Value related_value = pair_data.second;
      if (upstream_node->downstream.find(fused_up_node) != upstream_node->downstream.end()){
        upstream_node->downstream.erase(fused_up_node);
      }
      if (upstream_node->downstream.find(fused_down_node) != upstream_node->downstream.end()){
        upstream_node->downstream.erase(fused_down_node);
      }
      upstream_node->downstream[this] = related_value;
    }

    for (const auto& pair_data: downstream){
      FusionNode* downstream_node = pair_data.first;
      ::pir::Value related_value = pair_data.second;
      if (downstream_node->upstream.find(fused_up_node) != downstream_node->upstream.end()){
        downstream_node->upstream.erase(fused_up_node);
      }
      if (downstream_node->upstream.find(fused_down_node) != downstream_node->upstream.end()){
        downstream_node->upstream.erase(fused_down_node);
      }
      downstream_node->upstream[this] = related_value;
    }
  }

};

struct FusionGraph {

  explicit FusionGraph(
      const std::vector<::pir::Operation*>& ops,
      const std::vector<ir::Expr>& op_compute_bodies){

    // shardable_axes_ = InferShardableAxes(ops);
    VLOG(4) << "CreateFusionGraph";

    const auto& op_patterns = GetOpPatternKindVector(ops);
    CheckFusionInputValid(op_compute_bodies, op_patterns);

    std::unordered_map<::pir::Operation*, FusionNode*> op_to_node_map;

    for (int i=0; i<ops.size(); ++i){
      FusionNode* node = new FusionNode(op_compute_bodies[i], op_patterns[i]);
      op_to_node_map[ops[i]] = node;
      all_fusion_nodes_.emplace(node);
      node->expr_related_op = ops[i];
    }

    for (::pir::Operation* op : ops){
      FusionNode* cur_node = op_to_node_map[op];

      // add upstream nodes
      for (int i = 0; i < op->num_operands(); ++i){
        ::pir::Value related_value = op->operand_source(i);
        ::pir::Operation* input_op = related_value.defining_op();
        if (op_to_node_map.find(input_op) != op_to_node_map.end()){
          FusionNode* upstream_node = op_to_node_map[input_op];
          cur_node->upstream[upstream_node] = related_value;
          upstream_node->downstream[cur_node] = related_value;
        }
      }

      // add downstream nodes
      for (int i = 0; i < op->num_results(); ++i) {
        ::pir::Value related_value = op->result(i);
        for (auto consumer_it = related_value.use_begin(); consumer_it != related_value.use_end(); ++consumer_it) {
          ::pir::Operation* output_op = consumer_it->owner();
          if (op_to_node_map.find(output_op) != op_to_node_map.end()){
            FusionNode* downstream_node = op_to_node_map[output_op];
            cur_node->downstream[downstream_node]= related_value;
            downstream_node->upstream[cur_node] = related_value;
          }
        }
      }

      if (cur_node->upstream.size() == 0){
        entrance_nodes_.emplace(cur_node);
      }

      if (cur_node->downstream.size() == 0){
        exit_nodes_.emplace(cur_node);
      }
    }

    VLOG(4) << "FusionGraph Created, fusion node size: " << all_fusion_nodes_.size();
  }

  ~FusionGraph(){
    for (FusionNode* node: all_fusion_nodes_){
      delete node;
    }
  }

  std::vector<ir::Expr> DoFusion(){
    fuse_trivial_node();
    return get_expr_results();
  }

private:
  FusionNode* find_trivial_node(){
    for (FusionNode* node: all_fusion_nodes_){
      if (IsTrivialKind(node->op_pattern) && node->downstream.size() > 0){
        CHECK(node->op_compute_body.size() == 1);
        return node;
      }
    }
    return nullptr;
  }

  void fuse_trivial_node(){
    FusionNode* upstream;
    while((upstream = find_trivial_node()) != nullptr){
      std::unordered_map<FusionNode*, ::pir::Value> fusion_candidate = upstream->downstream;
      upstream->downstream.clear();
      for (const auto& pair_data : fusion_candidate) {
        FusionNode* downstream = pair_data.first;
        CHECK(downstream->op_compute_body.size() == 1);

        FusionNode* new_node;
        if (IsTrivialKind(downstream->op_pattern)){
          new_node = new FusionNode(
            TTFusion(upstream->op_compute_body[0], downstream->op_compute_body[0]),
            downstream->op_pattern
          );
        }else{
          new_node = new FusionNode(
            TRFusion(upstream->op_compute_body[0], downstream->op_compute_body[0]),
            downstream->op_pattern
          );
        }

        new_node->replace_topo_structure_of_fused_nodes(upstream, downstream);
        append_fusion_node(new_node);
        remove_fusion_node(downstream);
      }
      remove_fusion_node(upstream);
    }
  }

  std::vector<ir::Expr> get_expr_results() {
    std::vector<ir::Expr> output_exprs;
    for (const auto& node : all_fusion_nodes_) {
      output_exprs.insert(output_exprs.end(), node->op_compute_body.begin(), node->op_compute_body.end());
    }
    return output_exprs;
  }

  void remove_fusion_node(FusionNode* node){
    if (all_fusion_nodes_.find(node) != all_fusion_nodes_.end()){
      all_fusion_nodes_.erase(node);
    }
    if (entrance_nodes_.find(node) != entrance_nodes_.end()){
      entrance_nodes_.erase(node);
    }
    if (exit_nodes_.find(node) != exit_nodes_.end()){
      exit_nodes_.erase(node);
    }
    delete node;
  }

  void append_fusion_node(FusionNode* node){
    all_fusion_nodes_.emplace(node);
    if (node->upstream.size() == 0){
      entrance_nodes_.emplace(node);
    }

    if (node->downstream.size() == 0){
      exit_nodes_.emplace(node);
    }
  }

private:
  std::unordered_set<FusionNode*> all_fusion_nodes_;
  std::unordered_set<FusionNode*> entrance_nodes_;
  std::unordered_set<FusionNode*> exit_nodes_;

  // std::unordered_map<::pir::Value, ShardableAxes> shardable_axes_;
};

std::vector<FusionNode> ConstructFusionNodeElementwisely(
    const std::vector<ir::Expr>& op_compute_bodies,
    const std::vector<OpPatternKind>& op_kinds) {
  std::vector<FusionNode> output_vector;
  for (int i = 0; i < op_compute_bodies.size(); i++) {
    output_vector.emplace_back(op_compute_bodies[i], op_kinds[i]);
  }
  return output_vector;
}

bool IsAdjecentInjectiveBetween(const FusionNode& upstream_node,
                                const FusionNode& downstream_node) {
  return upstream_node.op_compute_body != downstream_node.op_compute_body &&
         IsTrivialKind(upstream_node.op_pattern) &&
         IsTrivialKind(downstream_node.op_pattern) &&
         IsAdjecent(upstream_node.op_compute_body[0],
                    downstream_node.op_compute_body[0]);
}

std::optional<FusionNode> FindUpstreamNodeUsedByOthers(
    const std::vector<FusionNode>& fusion_nodes) {
  for (int i = 0; i < fusion_nodes.size(); i++) {
    for (int j = i + 1; j < fusion_nodes.size(); j++) {
      if (IsAdjecentInjectiveBetween(fusion_nodes[i], fusion_nodes[j])) {
        return fusion_nodes[i];
      }
    }
  }
  return {};
}

std::vector<FusionNode> FuseEachUpstreamUse(
    const std::vector<FusionNode>& origin_nodes,
    const FusionNode& upstream_node) {
  std::vector<FusionNode> fused_nodes;
  std::transform(
      origin_nodes.begin(),
      origin_nodes.end(),
      std::back_inserter(fused_nodes),
      [&](const FusionNode& downstream_node) {
        if (IsAdjecentInjectiveBetween(upstream_node, downstream_node)) {
          return FusionNode(TTFusion(upstream_node.op_compute_body[0],
                                          downstream_node.op_compute_body[0]),
                            OpPatternKind::kInjective);
        }
        return downstream_node;
      });
  return fused_nodes;
}

std::vector<FusionNode> RemoveUpstreamTrivial(
    const FusionNode& upstream_node,
    const std::vector<FusionNode>& fusion_nodes) {
  auto removed_nodes = fusion_nodes;
  auto offset = std::find_if(fusion_nodes.begin(),
                             fusion_nodes.end(),
                             [&](const FusionNode& node) {
                               return node.op_compute_body ==
                                      upstream_node.op_compute_body;
                             }) -
                fusion_nodes.begin();
  removed_nodes.erase(removed_nodes.begin() + offset);
  return removed_nodes;
}

std::vector<FusionNode> FuseSingleUpstreamNode(
    const FusionNode& fusable_upstream,
    const std::vector<FusionNode>& fusion_nodes) {
  const auto& fused_node = FuseEachUpstreamUse(
      RemoveUpstreamTrivial(fusable_upstream, fusion_nodes), fusable_upstream);
  return fused_node;
}

std::vector<ir::Expr> ExtractBodiesFromFusionNodes(
    const std::vector<FusionNode>& fusion_nodes) {
  std::vector<ir::Expr> output_exprs;
  for (const auto& node : fusion_nodes) {
    output_exprs.emplace_back(node.op_compute_body[0]);
  }
  return output_exprs;
}

}  // namespace trivial_fusion_detail

std::vector<ir::Expr> TrivialOpFusion(
    const std::vector<::pir::Operation*>& ops,
    const std::vector<ir::Expr>& op_compute_bodies) {
  trivial_fusion_detail::FusionGraph graph = trivial_fusion_detail::FusionGraph(ops, op_compute_bodies);
  auto output = graph.DoFusion();
  VLOG(4) << "Fusion Result: output size is " << output.size();
  for (const auto& expr : output){
    VLOG(4) << expr;
  }
  return output;
}

// std::vector<ir::Expr> TrivialOpFusion_(
//     const std::vector<::pir::Operation*>& ops,
//     const std::vector<ir::Expr>& op_compute_bodies) {
//   const auto& op_patterns = trivial_fusion_detail::GetOpPatternKindVector(ops);
//   trivial_fusion_detail::CheckFusionInputValid(op_compute_bodies, op_patterns);
//   const auto& before_fused_nodes =
//       trivial_fusion_detail::ConstructFusionNodeElementwisely(op_compute_bodies,
//                                                               op_patterns);

//   auto fused_nodes_each_step = before_fused_nodes;
//   while (const auto& fusable_upstream =
//              trivial_fusion_detail::FindUpstreamNodeUsedByOthers(
//                  fused_nodes_each_step)) {
//     fused_nodes_each_step = trivial_fusion_detail::FuseSingleUpstreamNode(
//         fusable_upstream.value(), fused_nodes_each_step);
//   }

//   return trivial_fusion_detail::ExtractBodiesFromFusionNodes(
//       fused_nodes_each_step);
// }

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
