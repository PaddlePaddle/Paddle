// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/generate_map_expr.h"

#include "paddle/cinn/adt/anchor_sd_equation_context.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/graph_symbolic_dim_infer_ctx.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/naive_bidirection_equation_generator.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/cinn/adt/union_find.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/shape/utils/shape_optimization_utils.h"

#include "glog/logging.h"

PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_map_expr_enable_dynamic_shape);

namespace cinn::adt {

template <>
struct TreeMerger<Stmt> {
  using TreeT = Stmt;
  using tree_type = TreeT;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;

  using inner_data_type = typename inner_type::value_type;
  std::function<inner_data_type(const leaf_type&)> GetInnerDataForLeaf;

  inner_type MakeInnerNode(const inner_data_type& inner_data,
                           const List<TreeT>& children) const {
    return MapStmt<Stmt>{inner_data, children};
  }

  using MergeResult = std::tuple<tCommon<inner_data_type>,
                                 tLhsRemainder<inner_data_type>,
                                 tRhsRemainder<inner_data_type>>;

  MergeResult MergeInnerValue(const inner_data_type& lhs,
                              const inner_data_type& rhs) const {
    inner_data_type common{};
    inner_data_type lhs_remainder{};
    inner_data_type rhs_remainder{};
    int min_size = std::min(lhs->size(), rhs->size());
    int idx = 0;
    for (; idx < min_size; ++idx) {
      if (lhs->at(idx) == rhs->at(idx)) {
        common->emplace_back(lhs->at(idx));
      } else {
        break;
      }
    }
    for (int lhs_idx = idx; lhs_idx < lhs->size(); ++lhs_idx) {
      lhs_remainder->emplace_back(lhs->at(lhs_idx));
    }
    for (int rhs_idx = idx; rhs_idx < rhs->size(); ++rhs_idx) {
      rhs_remainder->emplace_back(rhs->at(rhs_idx));
    }
    return MergeResult{common, lhs_remainder, rhs_remainder};
  }
};

namespace {

using LoopDescriptor4IterVarT = std::function<LoopDescriptor(const Iterator&)>;

using FakeOpPlaceHolders = List<FakeOpPlaceHolder>;

Op MakeOp(const ::pir::Operation* op) { return {op}; }

template <typename DoEachT>
void VisitEachInputTensor(const ::pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_operands(); ++i) {
    DoEach(op->operand_source(i));
  }
}

bool HasDynamicShape(const ::pir::Value& tensor) {
  const auto& shape = hlir::framework::pir::CompatibleInfo::ValueShape(tensor);
  for (int dim : shape) {
    if (dim < 0) {
      CHECK_EQ(dim, -1);
      return true;
    }
  }
  return false;
}

List<Arg> MakeOpStmtInputList(const ::pir::Operation* op,
                              const hlir::framework::pir::Group* group) {
  List<Arg> ret{};

  VisitEachInputTensor(op, [&](const ::pir::Value& tensor) {
    if (HasDynamicShape(tensor)) {
      ret->emplace_back(adapter::DynamicTensor{tensor, group});
    } else {
      ret->emplace_back(adapter::Tensor{tensor});
    }
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOutputTensor(const ::pir::Operation* op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op->num_results(); ++i) {
    DoEach(const_cast<::pir::Operation*>(op)->result(i));
  }
}

List<Arg> MakeOpStmtOutputList(const ::pir::Operation* op,
                               const hlir::framework::pir::Group* group) {
  List<Arg> ret{};

  VisitEachOutputTensor(op, [&](const ::pir::Value& tensor) {
    if (HasDynamicShape(tensor)) {
      ret->emplace_back(adapter::DynamicTensor{tensor, group});
    } else {
      ret->emplace_back(adapter::Tensor{tensor});
    }
  });

  return ret;
}

template <typename DoEachT>
void VisitEachOpStmt(const std::shared_ptr<hlir::framework::pir::Group>& group,
                     const DoEachT& DoEach) {
  for (const auto* op : group->CollectOps()) {
    DoEach(OpStmt{MakeOp(op),
                  MakeOpStmtInputList(op, group.get()),
                  MakeOpStmtOutputList(op, group.get())});
  }
}

hlir::framework::OpPatternKind GetOpPatternKind(const ::pir::Operation* node) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*node);
}

bool CollectRewritedReductionOpStmts(const OpStmt& op_stmt, List<OpStmt>* ret) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  CHECK(op.Has<const ::pir::Operation*>());
  if (GetOpPatternKind(op.Get<const ::pir::Operation*>()) ==
      hlir::framework::OpPatternKind::kReduction) {
    tReduceInit<const ::pir::Operation*> init_op{
        op.Get<const ::pir::Operation*>()};
    (*ret)->emplace_back(OpStmt{init_op, List<Arg>{}, outputs});

    tReduceAcc<const ::pir::Operation*> acc_op{
        op.Get<const ::pir::Operation*>()};
    (*ret)->emplace_back(OpStmt{acc_op, inputs, outputs});
    return true;
  } else {
    return false;
  }
}

void CollectRewritedOpStmts(const OpStmt& op_stmt, List<OpStmt>* ret) {
  if (CollectRewritedReductionOpStmts(op_stmt, ret)) {
    return;
  }
  (*ret)->emplace_back(op_stmt);
}

List<OpStmt> MakeOpStmts(
    const std::shared_ptr<hlir::framework::pir::Group>& group) {
  List<OpStmt> ret{};

  VisitEachOpStmt(group, [&](const auto& op_stmt) {
    CollectRewritedOpStmts(op_stmt, &ret);
  });

  return ret;
}

template <typename DoEachT>
void PartitionIGroupOpStmts(const List<OpStmt>& op_stmts,
                            const DoEachT& DoEach) {
  const auto& EquationCtx4OpStmt =
      config::GenerateContext4LocalOpStmt(op_stmts);
  auto direction_equation_generator =
      std::make_shared<NaiveBidirectionEquationGenerator>(op_stmts,
                                                          EquationCtx4OpStmt);
  const auto& igroup_specs = PartitionOpStmts(
      EquationCtx4OpStmt, op_stmts, direction_equation_generator);
  for (const auto& igroup_spec : igroup_specs) {
    DoEach(igroup_spec);
  }
}

std::shared_ptr<IGroup> MakeIGroup(const AnchorGroup& igroup_spec) {
  std::shared_ptr<DirectionEquationGenerator> direction_equation_generator{
      new NaiveBidirectionEquationGenerator{igroup_spec.op_stmts,
                                            igroup_spec.EquationCtx4OpStmt}};
  CheckEquationSolvable(igroup_spec, direction_equation_generator);
  return std::make_shared<IGroup>(igroup_spec.op_stmts,
                                  igroup_spec.anchor_index,
                                  igroup_spec.EquationCtx4OpStmt);
}

std::vector<std::shared_ptr<IGroup>> GenerateIGroups(
    const std::shared_ptr<hlir::framework::pir::Group>& group) {
  std::vector<std::shared_ptr<IGroup>> ret{};

  List<OpStmt> op_stmts = MakeOpStmts(group);
  CHECK(!op_stmts->empty());

  PartitionIGroupOpStmts(op_stmts, [&](const auto& igroup_spec) {
    ret.push_back(MakeIGroup(igroup_spec));
  });

  return ret;
}

GraphView GenerateSdEquationGraphView(const std::shared_ptr<IGroup>& igroup,
                                      const ScheduleMesh& sched_mesh) {
  config::AnchorSdEquationContext ctx{sched_mesh, igroup->anchor_index()};
  igroup->set_anchor_sd_equation_ctx(ctx);

  Equations equations = igroup->anchor_sd_equation_ctx().value().equations();

  return Graph<Variable, Equation>::New(equations)->GetGraphView();
}

using TensorIndexExpr = Value;

std::unordered_map<Variable, const Value> MakeSdIterator2Iterator(
    const IGroup& igroup) {
  std::unordered_map<Variable, const Value> ret{};

  for (std::size_t i = 0; i < igroup.loop_iterators()->size(); ++i) {
    CHECK(ret.emplace(igroup.loop_iterators()->at(i),
                      igroup.loop_iterators()->at(i))
              .second);
  }

  return ret;
}

std::function<TensorIndexExpr(const Tensor&)> MakeGetterTensorIndexExpr(
    const std::shared_ptr<IndexExprInferContext>& ctx,
    const std::shared_ptr<IGroup>& igroup) {
  return [ctx, igroup](const Tensor& tensor) {
    // All indexes of same tensor have the same Value.
    const auto& indexes = igroup->GetIndexes(tensor);
    CHECK(!indexes.empty());
    return ctx->GetValue(indexes.at(0));
  };
}

TensorIteratorExpr4TensorT MakeGetterTensorIteratorExpr4Tensor(
    const std::shared_ptr<IndexExprInferContext>& ctx,
    const std::shared_ptr<IGroup>& igroup) {
  return [ctx, igroup](const Tensor& tensor) -> List<TensorIteratorExpr> {
    CHECK(!igroup->GetIndexes(tensor).empty());
    const auto& iterators = igroup->GetTensorIterators(tensor);
    List<TensorIteratorExpr> ret{};
    for (const auto& iterator : *iterators) {
      ret->emplace_back(ctx->GetValue(iterator));
    }
    return ret;
  };
}

LoopDescriptor4IterVarT MakeGetterLoopDescriptor4IterVar(
    const LoopIterators& loop_iters, const LoopDescriptors& sd) {
  CHECK_EQ(loop_iters->size(), sd->size());
  using Cache = std::unordered_map<Iterator, LoopDescriptor>;
  const auto& sd_iter2sd = std::make_shared<Cache>();
  for (std::size_t i = 0; i < loop_iters->size(); ++i) {
    CHECK(sd_iter2sd->emplace(loop_iters->at(i), sd->at(i)).second);
  }
  return [sd_iter2sd](const auto& sd_iter) { return sd_iter2sd->at(sd_iter); };
}

TreeMerger<Stmt> MakeTreeMerger(const MapIr& map_ir) {
  using Cache = std::unordered_map<OpStmt, LoopIterators>;
  auto cache = std::make_shared<Cache>();
  for (const auto& op_stmt : *(map_ir.op_stmts())) {
    CHECK(cache->emplace(op_stmt, map_ir.loop_iterators()).second);
  }

  TreeMerger<Stmt> tree_merger{};
  tree_merger.GetInnerDataForLeaf =
      ([=](const OpStmt& op_stmt) -> LoopIterators {
        return cache->at(op_stmt);
      });
  return tree_merger;
}

MapStmt<Stmt> MakeMapStmt(const MapIrList& map_irs) {
  List<Stmt> stmts{};
  for (const auto& map_ir : *map_irs) {
    const TreeMerger<Stmt>& tree_merger = MakeTreeMerger(map_ir);
    MergeTrees(tree_merger, &stmts, map_ir.op_stmts());
  }
  CHECK_EQ(stmts->size(), 1);
  CHECK(stmts->at(0).Has<MapStmt<Stmt>>());
  return stmts->at(0).Get<MapStmt<Stmt>>();
}

Tensor GetAnchorTensor(const std::shared_ptr<IGroup>& igroup) {
  return igroup->anchor_tensor();
}

template <typename DoEachT>
void VisitInputTensor(const hlir::framework::pir::Group& group,
                      const DoEachT& DoEach) {
  for (const ::pir::Value& node_data : group.GetInputOpValues()) {
    DoEach(node_data);
  }
}

template <typename DoEachT>
void VisitOutputTensor(const hlir::framework::pir::Group& group,
                       const DoEachT& DoEach) {
  for (const ::pir::Value& node_data : group.GetOutputOpValues()) {
    DoEach(node_data);
  }
}

List<Tensor> MakeInputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitInputTensor(*kgroup->cinn_group(), [&](const ::pir::Value& node_data) {
    ret->emplace_back(adapter::Tensor{node_data});
  });
  return ret;
}

List<Tensor> MakeOutputTensors(const std::shared_ptr<KGroup>& kgroup) {
  List<Tensor> ret{};
  VisitOutputTensor(*kgroup->cinn_group(), [&](const ::pir::Value& node_data) {
    ret->emplace_back(adapter::Tensor{node_data});
  });
  return ret;
}

std::unordered_map<Tensor, std::unordered_set<std::shared_ptr<IGroup>>>
CollectInterfaceTensors(const std::vector<std::shared_ptr<IGroup>>& igroups) {
  std::unordered_map<Tensor, std::unordered_set<std::shared_ptr<IGroup>>>
      tensor2igroups{};
  for (const auto& igroup : igroups) {
    igroup->VisitEachOpStmt([&](const OpStmt& op_stmt) {
      const auto& [_, inputs, outputs] = op_stmt.tuple();
      for (const auto& tensor : *inputs.value()) {
        tensor2igroups[tensor].insert(igroup);
      }
      for (const auto& tensor : *outputs.value()) {
        tensor2igroups[tensor].insert(igroup);
      }
    });
  }
  std::unordered_map<Tensor, std::unordered_set<std::shared_ptr<IGroup>>> ret{};
  using TensorIGroupPair =
      std::pair<Tensor, std::unordered_set<std::shared_ptr<IGroup>>>;
  std::copy_if(
      tensor2igroups.begin(),
      tensor2igroups.end(),
      std::inserter(ret, ret.end()),
      [](const TensorIGroupPair& pair) { return pair.second.size() >= 2; });
  return ret;
}

std::shared_ptr<IndexExprInferContext> SolveIGroupThenReturnCtx(
    const std::shared_ptr<IGroup>& igroup) {
  GraphView igroup_view = igroup->GetDefaultGraphView();

  std::unordered_map<Variable, const Value> infer_map{};
  std::vector<Variable> infer_start{};
  for (const auto& iterator : *igroup->GetAnchorIterators()) {
    CHECK(infer_map.emplace(iterator, iterator).second);
    infer_start.emplace_back(iterator);
  }

  auto ctx = std::make_shared<IndexExprInferContext>(infer_map);
  SolveEquations(igroup_view, infer_start, ctx.get());
  return ctx;
}

using InferContext4IGroupT =
    std::function<const std::shared_ptr<IndexExprInferContext>(
        const std::shared_ptr<IGroup>)>;
InferContext4IGroupT MakeGetterInferContext4IGroup(
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  std::unordered_map<std::shared_ptr<IGroup>,
                     std::shared_ptr<IndexExprInferContext>>
      ret{};
  for (const auto& igroup : igroups) {
    CHECK(ret.emplace(igroup, SolveIGroupThenReturnCtx(igroup)).second);
  }
  return [ret](const std::shared_ptr<IGroup>& igroup)
             -> const std::shared_ptr<IndexExprInferContext> {
    CHECK_GT(ret.count(igroup), 0);
    return ret.at(igroup);
  };
}

List<Value> CollectTensorIteratorExpr(
    const Tensor& tensor,
    const std::shared_ptr<IGroup>& igroup,
    const InferContext4IGroupT& InferContext4IGroup) {
  List<Value> ret{};
  const auto& ctx = InferContext4IGroup(igroup);
  for (const auto& iterator : *igroup->GetTensorIterators(tensor)) {
    ret->emplace_back(ctx->GetValue(iterator));
  }
  return ret;
}

bool ValueIsomorphicImpl(const Undefined& lhs, const Undefined& rhs) {
  LOG(FATAL) << "Dead code";
}
bool ValueIsomorphicImpl(const Ok&, const Ok&) { LOG(FATAL) << "Dead code"; }
bool ValueIsomorphicImpl(const Iterator&, const Iterator&) { return true; }
bool ValueIsomorphicImpl(const DimExpr&, const DimExpr&) { return false; }
bool ValueIsomorphicImpl(const List<Value>&, const List<Value>&) {
  LOG(FATAL) << "Not Implement yet!";
}
bool ValueIsomorphicImpl(const IndexDotValue<Value, List<DimExpr>>&,
                         const IndexDotValue<Value, List<DimExpr>>&) {
  LOG(FATAL) << "Not Implement yet!";
}
bool ValueIsomorphicImpl(const IndexUnDotValue<Value, List<DimExpr>>&,
                         const IndexUnDotValue<Value, List<DimExpr>>&) {
  LOG(FATAL) << "Not Implement yet!";
}
bool ValueIsomorphicImpl(const ListGetItem<Value, DimExpr>&,
                         const ListGetItem<Value, DimExpr>&) {
  LOG(FATAL) << "Not Implement yet!";
}
bool ValueIsomorphicImpl(const BroadcastedIterator<Value, DimExpr>&,
                         const BroadcastedIterator<Value, DimExpr>&) {
  LOG(FATAL) << "Not Implement yet!";
}
bool ValueIsomorphicImpl(const PtrGetItem<Value>&, const PtrGetItem<Value>&) {
  LOG(FATAL) << "Not Implement yet!";
}

bool ValueIsomorphic(const Value& lhs, const Value& rhs) {
  return std::visit(
      [](const auto& lhs, const auto& rhs) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return ValueIsomorphicImpl(lhs, rhs);
        } else {
          return false;
        }
      },
      lhs.variant(),
      rhs.variant());
}

void UpdateUnionFindByInterfaceTensor(
    const Tensor& interface_tensor,
    const std::unordered_set<std::shared_ptr<IGroup>>& igroups,
    const InferContext4IGroupT& InferContext4IGroup,
    UnionFind<Iterator>* uf) {
  CHECK_GE(igroups.size(), 2);
  const auto& lhs_iter_exprs = CollectTensorIteratorExpr(
      interface_tensor, *igroups.begin(), InferContext4IGroup);
  for (const auto& expr : *lhs_iter_exprs) {
    VLOG(1) << "Left iterators: " << ToTxtString(expr);
  }
  for (const auto& igroup : igroups) {
    const auto& rhs_iter_exprs = CollectTensorIteratorExpr(
        interface_tensor, igroup, InferContext4IGroup);
    CHECK_EQ(lhs_iter_exprs->size(), rhs_iter_exprs->size());
    for (const auto& expr : *rhs_iter_exprs) {
      VLOG(1) << "Right iterators: " << ToTxtString(expr);
    }

    for (std::size_t i = 0; i < lhs_iter_exprs->size(); ++i) {
      if (ValueIsomorphic(lhs_iter_exprs->at(i), rhs_iter_exprs->at(i))) {
        CHECK(lhs_iter_exprs->at(i).Has<Iterator>() &&
              rhs_iter_exprs->at(i).Has<Iterator>());
        uf->Union(lhs_iter_exprs->at(i).Get<Iterator>(),
                  rhs_iter_exprs->at(i).Get<Iterator>());
      } else {
        // Do nothing
      }
    }
  }
}

std::shared_ptr<UnionFind<Iterator>> CreateAnchorIteratorUf(
    const std::vector<std::shared_ptr<IGroup>>& igroups,
    std::unordered_map<Tensor, std::unordered_set<std::shared_ptr<IGroup>>>&
        interface_tensor2igroups,
    const InferContext4IGroupT& InferContext4IGroup) {
  auto uf = std::make_shared<UnionFind<Iterator>>();
  for (const auto& [tensor, interface_igroups] : interface_tensor2igroups) {
    UpdateUnionFindByInterfaceTensor(
        tensor, interface_igroups, InferContext4IGroup, uf.get());
  }
  return uf;
}

using ShardableDimAndPerm = std::pair<List<ScheduleDim>, List<int>>;

List<int> GenerateAnchorIteratorPerm(
    const List<Iterator>& igroup_iterators,
    const List<Iterator>& shardable_igroup_iterators,
    const std::shared_ptr<UnionFind<Iterator>>& uf) {
  const auto& FindIteratorPerm =
      [&](const Iterator& shardable_iterator) -> int {
    for (std::size_t i = 0; i < igroup_iterators->size(); ++i) {
      if (uf->IsConnected(igroup_iterators->at(i), shardable_iterator)) {
        return i;
      } else {
        // Do nothing
      }
    }
    LOG(FATAL) << "Iterator not found";
  };
  List<int> shardable_idx{};
  for (const auto& shardable_iterator : *shardable_igroup_iterators) {
    shardable_idx->push_back(FindIteratorPerm(shardable_iterator));
  }
  return shardable_idx;
}

List<ShardableDimAndPerm> GenerateShardableDimWithOrder(
    const std::vector<std::shared_ptr<IGroup>>& igroups,
    const List<Iterator>& shardable_igroup_iterators,
    const std::shared_ptr<UnionFind<Iterator>>& uf) {
  List<ShardableDimAndPerm> ret{};
  for (const auto& igroup : igroups) {
    ret->emplace_back(std::make_pair(
        igroup->anchor_schedule_dims(),
        GenerateAnchorIteratorPerm(
            igroup->GetAnchorIterators(), shardable_igroup_iterators, uf)));
  }
  return ret;
}

List<Iterator> GetIntersectionShardableIterators(
    const List<Iterator>& lhs,
    const List<Iterator>& rhs,
    const std::shared_ptr<UnionFind<Iterator>>& uf) {
  List<Iterator> ret{};
  for (const auto& lhs_iterator : *lhs) {
    for (const auto& rhs_iterator : *rhs) {
      if (uf->IsConnected(lhs_iterator, rhs_iterator)) {
        ret->push_back(lhs_iterator);
      } else {
        // Do nothing
      }
    }
  }
  return ret;
}

List<Iterator> FilterShardableIterators(
    const std::vector<std::shared_ptr<IGroup>>& igroups,
    const std::unordered_map<Tensor,
                             std::unordered_set<std::shared_ptr<IGroup>>>&
        interface_tensor2igroups,
    const std::shared_ptr<UnionFind<Iterator>>& uf,
    const InferContext4IGroupT& InferContext4IGroup) {
  const auto& FilterIterators =
      [&](const List<Value>& iterator_exprs) -> List<Iterator> {
    List<Iterator> ret{};
    for (const auto& iterator_expr : *iterator_exprs) {
      if (iterator_expr.Has<Iterator>() &&
          uf->NodeCluster(iterator_expr.Get<Iterator>()).size() ==
              igroups.size()) {
        ret->push_back(iterator_expr.Get<Iterator>());
      }
    }
    return ret;
  };

  std::optional<List<Iterator>> opt_ret{std::nullopt};
  for (const auto& [tensor, interface_igroups] : interface_tensor2igroups) {
    const auto& interface_igroup = *interface_igroups.begin();
    const auto& iterator_exprs = CollectTensorIteratorExpr(
        tensor, interface_igroup, InferContext4IGroup);
    List<Iterator> shardable_iterators = FilterIterators(iterator_exprs);
    if (opt_ret.has_value()) {
      opt_ret = GetIntersectionShardableIterators(
          opt_ret.value(), shardable_iterators, uf);
    } else {
      opt_ret = shardable_iterators;
    }
  }
  if (opt_ret.has_value()) {
    return opt_ret.value();
  } else {
    return List<Iterator>{};
  }
}

List<ShardableDimAndPerm> CollectShardableScheduleDims(
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  const auto& InferContext4IGroup = MakeGetterInferContext4IGroup(igroups);

  std::unordered_map<Tensor, std::unordered_set<std::shared_ptr<IGroup>>>
      interface_tensor2igroups = CollectInterfaceTensors(igroups);
  const auto& iterator_uf = CreateAnchorIteratorUf(
      igroups, interface_tensor2igroups, InferContext4IGroup);

  const auto& shardable_igroup_iterators = FilterShardableIterators(
      igroups, interface_tensor2igroups, iterator_uf, InferContext4IGroup);

  return GenerateShardableDimWithOrder(
      igroups, shardable_igroup_iterators, iterator_uf);
}

List<Iterator> MakeSoleIGroupScheduleIterator(
    const std::shared_ptr<IGroup>& igroup, const ScheduleMesh& sched_mesh) {
  config::AnchorSdEquationContext ctx{sched_mesh, igroup->anchor_index()};
  igroup->set_anchor_sd_equation_ctx(ctx);
  return igroup->loop_iterators();
}

List<Iterator> CollectScheduleIterators(
    const std::vector<std::shared_ptr<IGroup>>& igroups,
    const List<ScheduleMesh>& sched_meshs,
    int shardable_prefix_size) {
  List<Iterator> ret{};
  const auto& first_igroup_iterators =
      MakeSoleIGroupScheduleIterator(igroups.at(0), sched_meshs->at(0));
  ret->insert(ret->end(),
              first_igroup_iterators->begin(),
              first_igroup_iterators->end());
  for (std::size_t i = 1; i < igroups.size(); ++i) {
    const auto& iterators =
        MakeSoleIGroupScheduleIterator(igroups.at(i), sched_meshs->at(i));
    ret->insert(ret->end(),
                iterators->begin() + shardable_prefix_size,
                iterators->end());
  }
  return ret;
}

std::unordered_map<Variable, const Value> MakeStartScheduleIteratorMap(
    const List<Iterator>& schedule_iterators) {
  std::unordered_map<Variable, const Value> ret{};
  for (const auto& iterator : *schedule_iterators) {
    CHECK(ret.emplace(iterator, iterator).second);
  }
  return ret;
}

GraphView CollectIGroupView(const std::shared_ptr<IGroup>& igroup) {
  const auto& opt_sd_equation_ctx = igroup->anchor_sd_equation_ctx();
  CHECK(opt_sd_equation_ctx.has_value());
  Equations equations = opt_sd_equation_ctx.value().equations();
  GraphView sd_view = Graph<Variable, Equation>::New(equations)->GetGraphView();
  return sd_view.Merge(igroup->GetDefaultGraphView());
}

GraphView GenerateShardableDimEquationView(
    const std::vector<std::shared_ptr<IGroup>>& igroups,
    int shardable_prefix_size) {
  Equations equations{};
  const auto& first_group_iterators = igroups.at(0)->loop_iterators();
  for (std::size_t i = 1; i < igroups.size(); ++i) {
    const auto& group_iterators = igroups.at(i)->loop_iterators();
    for (std::size_t j = 0; j < shardable_prefix_size; ++j) {
      equations->emplace_back(Identity<tOut<Iterator>, tIn<Iterator>>(
          first_group_iterators->at(j), group_iterators->at(j)));
      equations->emplace_back(Identity<tOut<Iterator>, tIn<Iterator>>(
          group_iterators->at(j), first_group_iterators->at(j)));
    }
  }
  return Graph<Variable, Equation>::New(equations)->GetGraphView();
}

std::shared_ptr<IndexExprInferContext> SolveEquationsThenReturnCtx(
    const std::shared_ptr<IGroup>& igroup,
    const GraphView shardable_equation_view,
    const std::unordered_map<Variable, const Value>& init_var2value) {
  GraphView igroup_view = CollectIGroupView(igroup);
  GraphView merged_view = igroup_view.Merge(shardable_equation_view);

  auto ctx = std::make_shared<IndexExprInferContext>(init_var2value);
  std::vector<Variable> infer_start{};
  for (const auto& [var, _] : init_var2value) {
    infer_start.emplace_back(var);
  }

  SolveEquations(merged_view, infer_start, ctx.get());
  return ctx;
}

int GetShardablePrefixSize(
    const List<ShardableDimAndPerm>& shardable_schedule_dims) {
  const auto& dim_perm = shardable_schedule_dims->at(0);
  const auto& [dim, perm] = dim_perm;
  return perm->size();
}

AnchoredMapStmt GenerateAnchoredMapStmt(
    const std::shared_ptr<IGroup>& igroup,
    const GraphView& shardable_equation_view,
    const List<Iterator>& schedule_iters,
    const LoopDescriptors& sd) {
  const auto& init_var2value = MakeStartScheduleIteratorMap(schedule_iters);
  const auto& ctx = SolveEquationsThenReturnCtx(
      igroup, shardable_equation_view, init_var2value);

  const auto& TensorIndexExpr4Tensor = MakeGetterTensorIndexExpr(ctx, igroup);
  const auto& map_irs = GenerateMapIrListForLoopFuse(
      igroup->op_stmts(), schedule_iters, TensorIndexExpr4Tensor);

  return AnchoredMapStmt{MakeMapStmt(map_irs),
                         TensorIndexExpr4Tensor,
                         MakeGetterTensorIteratorExpr4Tensor(ctx, igroup),
                         MakeGetterLoopDescriptor4IterVar(schedule_iters, sd)};
}

List<AnchoredMapStmt> GenerateAnchoredMapStmts(
    const std::vector<std::shared_ptr<IGroup>>& igroups) {
  const auto& shardable_schedule_dims = CollectShardableScheduleDims(igroups);
  const auto& [sched_meshs, loop_types] =
      CreateOptimizedScheduleMeshs(shardable_schedule_dims);

  const auto& sd = CreateScheduleDescriptor(
      sched_meshs, loop_types, GetShardablePrefixSize(shardable_schedule_dims));
  const auto& schedule_iters = CollectScheduleIterators(
      igroups, sched_meshs, GetShardablePrefixSize(shardable_schedule_dims));
  const auto& shardable_equation_view = GenerateShardableDimEquationView(
      igroups, GetShardablePrefixSize(shardable_schedule_dims));

  List<AnchoredMapStmt> ret{};
  for (std::size_t i = 0; i < igroups.size(); ++i) {
    ret->emplace_back(GenerateAnchoredMapStmt(
        igroups.at(i), shardable_equation_view, schedule_iters, sd));
  }
  return ret;
}

List<AnchoredMapStmt> MakeAnchoredMapStmts(
    const std::shared_ptr<KGroup>& kgroup) {
  return GenerateAnchoredMapStmts(kgroup->igroups());
}

MapExpr GenerateMapExpr(const std::shared_ptr<KGroup>& kgroup) {
  // MapExpr = Kernel;
  // Kernel = ([AnchoredMapStmt], In [Tensor], Out [Tensor])
  return MapExpr{MakeAnchoredMapStmts(kgroup),
                 MakeInputTensors(kgroup),
                 MakeOutputTensors(kgroup)};
}

}  // namespace

MapExpr GenerateMapExpr(
    const std::shared_ptr<hlir::framework::pir::Group>& group) {
  const auto& igroups = GenerateIGroups(group);

  const auto& kgroup = std::make_shared<KGroup>(group, igroups);

  return GenerateMapExpr(kgroup);
}

void TryGenerateMapExprFromGraph(
    const hlir::framework::pir::GroupList& groups) {
  if (!FLAGS_cinn_enable_map_expr) {
    return;
  }
  for (const auto& fusion_group : groups) {
    fusion_group->set_graph_symbolic_dim_infer_ctx(
        std::make_unique<config::GraphSymbolicDimInferCtx>(fusion_group.get()));
    const auto& map_expr = GenerateMapExpr(fusion_group);
    VLOG(4) << ToTxtString(map_expr, fusion_group->group_id);
    fusion_group->set_map_expr_ctx(std::make_shared<MapExprCtx>(
        map_expr,
        fusion_group->graph_symbolic_dim_infer_ctx()
            ->map_expr_symbolic2dialect_symbolic()));
  }
}

void TryGenerateMapExprFromGroup(
    const std::shared_ptr<hlir::framework::pir::Group>& fusion_group) {
  if (!FLAGS_cinn_enable_map_expr) {
    return;
  }
  fusion_group->set_graph_symbolic_dim_infer_ctx(
      std::make_unique<config::GraphSymbolicDimInferCtx>(fusion_group.get()));
  const auto& map_expr = GenerateMapExpr(fusion_group);
  VLOG(4) << "Generate MapExpr: \n"
          << ToTxtString(map_expr, fusion_group->group_id);
  fusion_group->set_map_expr_ctx(
      std::make_shared<MapExprCtx>(map_expr,
                                   fusion_group->graph_symbolic_dim_infer_ctx()
                                       ->map_expr_symbolic2dialect_symbolic()));
}

}  // namespace cinn::adt
