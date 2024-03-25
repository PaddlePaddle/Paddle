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
#pragma once

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
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& first,
                            const std::vector<T>& second) {
  std::vector<T> result = first;
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

std::vector<ir::Var> ExprVec2VarVec(const std::vector<ir::Expr>& in);
std::vector<ir::Expr> VarVec2ExprVec(const std::vector<ir::Var>& in);

std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Expr& body,
                                            const ir::Tensor& tensor);

struct MappingTargetExprToDestExprMutator : public ir::IRMutator<> {
  explicit MappingTargetExprToDestExprMutator(const ir::Expr& source,
                                              const ir::Expr& dest);

  void operator()(Expr* expr);

 private:
  void Visit(const ir::Load* load, Expr* op) override;
  void Visit(const ir::Store* store, Expr* op) override;
  void Visit(const ir::Reduce* reduce, Expr* op) override;

 private:
  ir::Expr source_;
  ir::Expr dest_;
};

bool CheckIterEq(const std::vector<ir::Var>& up_iter,
                 const std::vector<ir::Var>& down_iter);

ir::Expr CopyedReplaceExpr(const Expr& source,
                           const std::vector<Var>& replaced,
                           const std::vector<Expr>& candidates);
void SubstitudeTargetExprWithDestExpr(const ir::Expr& source,
                                      const ir::Expr& dest,
                                      ir::Expr* body);

ir::Expr SubstitudeIndexVector(const Expr& source,
                               const std::vector<Var>& load_vars,
                               const std::vector<ir::Expr>& indices);

template <typename FusionOp>
void ReplaceDownstreamLoadExprWithUpstreamComputeBody(
    const FusionOp& upstream,
    const ir::Expr& downstream_load_expr,
    ir::Expr* downstream_body) {
  ComposeUtils::SubstitudeTargetExprWithDestExpr(
      downstream_load_expr,
      ComposeUtils::SubstitudeIndexVector(
          GetComputeBody(upstream),
          GetOutputIters(upstream),
          downstream_load_expr.As<ir::Load>()->indices),
      downstream_body);
}
}  // namespace ComposeUtils

namespace SearchUtils {

using ExprSet = std::vector<ir::Expr>;
using Func = std::function<ExprSet(const ir::Expr& x)>;
struct Mapping {
  Func f_;
  std::string name;
  explicit Mapping(Func f, std::string s = "");

  ExprSet operator()(const ir::Expr& x) const;
  ir::Expr GetSingle(const ir::Expr& x) const;
  Mapping operator*(Mapping x) const;
  static Mapping GetIdentity();
};

template <typename Teller>
Mapping Collector(Teller t, std::string name = "") {
  return Mapping(
      [=](const ir::Expr& x) -> ExprSet {
        const auto& rs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(x, t);
        return std::vector(rs.begin(), rs.end());
      },
      name);
}

template <typename FilterFunc>
Mapping FilterMaker(FilterFunc t, std::string name) {
  return Mapping(
      [=](const ir::Expr& x) -> ExprSet {
        if (t(x)) {
          return {x};
        }
        return {};
      },
      name);
}

extern Mapping Identity;

extern Mapping Store2Value;

extern Mapping Realizer2ScheduleBlock;

extern Mapping ScheduleBlock2Body;

extern Mapping ScheduleBlockRealizeNotRoot;

extern Mapping ScheduleBlockRealizeIsNotInit;

extern Mapping ScheduleBlockRealizeIsInit;

extern Mapping IsFor;

extern Mapping ChildScheduleBlocks;

extern Mapping ChildScheduleBlockRealizes;

extern Mapping For2Min;

extern Mapping For2Max;

extern Mapping ChildStores;

extern Mapping ChildTensorLoads;

extern Mapping ChildTensorStores;

extern Mapping ChildFors;

Mapping IsForIterVar(const ir::Var& var);

Mapping FilterLoadByTensor(const ir::Tensor& tensor);

Mapping FindFather(const ir::Expr& root);

template <class T, class M>
std::vector<T> MapVector(const std::vector<T>& as, M func) {
  std::vector<T> res;
  for (const auto& a : as) {
    res.push_back(func(a));
  }
  return res;
}
}  // namespace SearchUtils

namespace TransformerUtils {
using TransformFunc = std::function<ir::Expr(ir::Expr)>;
struct Transformer {
  TransformFunc f_;
  explicit Transformer(TransformFunc f);
  ir::Expr operator()(const ir::Expr& x) const;
  Transformer operator*(const Transformer& x) const;
};

extern Transformer Identity;

Transformer WrapForTransformer(const ir::Var& v);

Transformer WrapForsTransformer(const std::vector<ir::Var>& vs);
Transformer ChangeTensorLoadTransformer(const ir::Tensor& tensor,
                                        const ir::Expr& dst_load);

void ReplaceTarget(ir::Expr* e, const ir::Expr& t, const ir::Expr dst);

Transformer WrapStoreTransformer(const ir::Tensor& tensor,
                                 const std::vector<ir::Expr>& indices);

Transformer WrapReduceOperation(const ir::Reduce::ReduceType& reduce_type,
                                const ir::Tensor& tensor,
                                const std::vector<ir::Expr>& axis_exprs);

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars);

Transformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                 const std::vector<ir::Var>& dest_vars);

Transformer SubstitudeByScheduleBlockRealize(const ir::Expr& realize);

Transformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                 const std::string& tensor_name);
}  // namespace TransformerUtils

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops);

template <class A, class C, class Func>
void SequenceMutator(const std::vector<A>& as, C* acc, const Func& mutator) {
  VLOG(4) << "SequenceTransform Init: " << acc;
  for (int i = 0; i < as.size(); ++i) {
    mutator(as[i], acc);
    VLOG(4) << "SequenceTransform Iter: " << acc;
  }
}

bool IsTrivialKind(OpPatternKind kind);

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns);

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
