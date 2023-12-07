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

#include "paddle/cinn/adt/graph_symbolic_dim_infer_ctx.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/arithmetic.h"
#include "paddle/cinn/adt/dim_expr_simplifier.h"
#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/symbolic_dim.h"
#include "paddle/cinn/adt/unique_id.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/shape/utils/shape_optimization_utils.h"

PD_DECLARE_bool(cinn_enable_map_expr_dynamic_shape);

namespace cinn::adt::config {

namespace {

// clang-format off
// Dim equations' variables:
//
// DimVar = ShapeDialectTensorDim | ShapeDialectTempDim
// ShapeDialectTensorDim = (Tensor, tAxis int)
// ShapeDialectTempDim = (tDimVar UniqueId)
//
// Dim equations' functions:
// DimFunction = DimIdentity (tOut ShapeDialectTensorDim)
//                           (tIn ShapeDialectTensorDim)
//             | DimProduct (tOut ShapeDialectTensorDim)
//                          [tIn DimVar]
//             | DimReciprocal (tOut ShapeDialectTempDim)
//                             (tIn ShapeDialectTensorDim)
//
// Dim equations' solutions:
//
//     DimExpr
// clang-format on

// ShapeDialectTensorDim = (::pir::Value, tAxis int)
struct ShapeDialectTensorDim {
  ::pir::Value tensor;
  int axis;

  bool operator==(const ShapeDialectTensorDim& other) const {
    return this->tensor == other.tensor && this->axis == other.axis;
  }
};

DEFINE_ADT_TAG(tDimVar);
using ShapeDialectTempDim = tDimVar<UniqueId>;

DEFINE_ADT_UNION(DimVar, ShapeDialectTensorDim, ShapeDialectTempDim);
OVERLOAD_OPERATOR_EQ_NE(DimVar, UnionEqual);

std::size_t GetHashValueImpl(const ShapeDialectTensorDim& dim) {
  return hash_combine(std::hash<::pir::Value>()(dim.tensor), dim.axis);
}

std::size_t GetHashValueImpl(const ShapeDialectTempDim& dim) {
  return dim.value().unique_id();
}

OVERRIDE_UNION_GET_HASH_VALUE(DimVar);

template <typename T0, typename T1>
struct DimIdentity;

// DimIdentity (tOut ShapeDialectTensorDim) (tIn ShapeDialectTensorDim)
template <>
struct DimIdentity<tOut<ShapeDialectTensorDim>, tIn<ShapeDialectTensorDim>>
    : public Tuple<tOut<ShapeDialectTensorDim>, tIn<ShapeDialectTensorDim>> {
  using Tuple<tOut<ShapeDialectTensorDim>, tIn<ShapeDialectTensorDim>>::Tuple;
};

template <typename T0, typename T1>
struct DimProduct;

// DimProduct (tOut ShapeDialectTensorDim) [tIn DimVar]
template <>
struct DimProduct<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>
    : public Tuple<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>> {
  using Tuple<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>::Tuple;
};

template <typename T0, typename T1>
struct DimReciprocal;

// DimReciprocal (tOut ShapeDialectTempDim) (tIn ShapeDialectTensorDim)
template <>
struct DimReciprocal<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>>
    : public Tuple<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>> {
  using Tuple<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>>::Tuple;
};

// clang-format off
// DimFunction = DimIdentity (tOut ShapeDialectTensorDim)
//                           (tIn ShapeDialectTensorDim)
//             | DimProduct (tOut ShapeDialectTensorDim)
//                          [tIn DimVar]
//             | DimReciprocal (tOut ShapeDialectTempDim)
//                             (tIn ShapeDialectTensorDim)
// clang-format on

DEFINE_ADT_UNION(
    DimFunction,
    DimIdentity<tOut<ShapeDialectTensorDim>, tIn<ShapeDialectTensorDim>>,
    DimProduct<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>,
    DimReciprocal<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>>);
}  // namespace

using DimFunctions = List<DimFunction>;

}  // namespace cinn::adt::config

namespace std {

template <>
struct hash<cinn::adt::config::ShapeDialectTensorDim> final {
  std::size_t operator()(
      const cinn::adt::config::ShapeDialectTensorDim& dim) const {
    return cinn::adt::hash_combine(std::hash<::pir::Value>()(dim.tensor),
                                   dim.axis);
  }
};

template <>
struct hash<cinn::adt::config::DimVar> final {
  std::size_t operator()(const cinn::adt::config::DimVar& dim) const {
    return cinn::adt::config::GetHashValue(dim);
  }
};

}  // namespace std

namespace cinn::adt {
template <>
struct GraphTrait<config::DimVar, config::DimFunction> {
  static std::pair<std::unordered_set<config::DimVar>,
                   std::unordered_set<config::DimVar>>
  CollectInputAndOutputVariables(const config::DimFunction& function) {
    using cinn::adt::config::DimVar;
    using cinn::adt::config::ShapeDialectTempDim;
    using cinn::adt::config::ShapeDialectTensorDim;

    std::unordered_set<DimVar> in_variables;
    std::unordered_set<DimVar> out_variables;

    function >>
        match{
            [&](const config::DimIdentity<tOut<ShapeDialectTensorDim>,
                                          tIn<ShapeDialectTensorDim>>&
                    identity) {
              const auto& [out_tensor_dim, in_tensor_dim] = identity.tuple();
              out_variables.emplace(DimVar{out_tensor_dim.value()});
              in_variables.emplace(DimVar{in_tensor_dim.value()});
            },
            [&](const config::DimProduct<tOut<ShapeDialectTensorDim>,
                                         tIn<List<DimVar>>>& dim_product) {
              const auto& [out_tensor_dim, in_dim_vars] = dim_product.tuple();
              out_variables.emplace(DimVar{out_tensor_dim.value()});
              for (const auto& in_dim_var : *(in_dim_vars.value())) {
                in_variables.emplace(in_dim_var);
              }
            },
            [&](const config::DimReciprocal<tOut<ShapeDialectTempDim>,
                                            tIn<ShapeDialectTensorDim>>&
                    reciprocal) {
              const auto& [out_temp_dim, in_tensor_dim] = reciprocal.tuple();
              out_variables.emplace(DimVar{out_temp_dim.value()});
              in_variables.emplace(DimVar{in_tensor_dim.value()});
            },
        };
    return std::make_pair(in_variables, out_variables);
  }
};
}  // namespace cinn::adt

namespace cinn::adt::config {

namespace {

template <typename DoEachT>
void VisitEachTensorPairOfOp(const ::pir::Operation* op_node,
                             const DoEachT& DoEach) {
  std::vector<::pir::Value> all_tensors{};
  for (const ::pir::Value& tensor : op_node->operands_source()) {
    all_tensors.emplace_back(tensor);
  }
  for (const ::pir::Value& tensor :
       const_cast<::pir::Operation*>(op_node)->results()) {
    all_tensors.emplace_back(tensor);
  }
  for (std::size_t i = 0; i < all_tensors.size(); ++i) {
    for (std::size_t j = i + 1; j < all_tensors.size(); ++j) {
      DoEach(all_tensors.at(i), all_tensors.at(j));
    }
  }
}

template <typename T, typename DoEachT>
void VisitEachIdxPairOfTwoVectors(const std::vector<T>& lhs,
                                  const std::vector<T>& rhs,
                                  const DoEachT& DoEach) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    for (std::size_t j = 0; j < rhs.size(); ++j) {
      DoEach(i, j);
    }
  }
}

List<ShapeDialectTensorDim> MakeShapeDialectTensorDimList(
    const ::pir::Value& tensor) {
  List<ShapeDialectTensorDim> ret{};
  for (std::size_t i = 0;
       i < hlir::framework::pir::CompatibleInfo::ValueShape(tensor).size();
       ++i) {
    ret->emplace_back(ShapeDialectTensorDim{tensor, i});
  }
  return ret;
}

List<ShapeDialectTempDim> GenerateReciprocalConstraints(
    const List<ShapeDialectTensorDim>& tensor_dims, DimFunctions* ret) {
  List<ShapeDialectTempDim> temp_dims{};
  for (const auto& tensor_dim : *tensor_dims) {
    ShapeDialectTempDim temp_dim{UniqueId::New()};
    (*ret)->emplace_back(
        DimReciprocal<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>>{
            temp_dim, tensor_dim});
    temp_dims->emplace_back(temp_dim);
  }
  return temp_dims;
}

// (a * b == c * d) => (a = c * d * 1/b)
List<DimVar> CollectProductDimVarExceptIdx(
    const List<ShapeDialectTensorDim>& tensor_dims,
    const List<ShapeDialectTempDim>& temp_dims,
    std::size_t ignore_idx) {
  CHECK_EQ(tensor_dims->size(), temp_dims->size());
  List<DimVar> ret{};
  for (const auto& tensor_dim : *tensor_dims) {
    ret->emplace_back(tensor_dim);
  }
  for (std::size_t i = 0; i < temp_dims->size(); ++i) {
    if (i == ignore_idx) {
      continue;
    }
    ret->emplace_back(temp_dims->at(i));
  }
  return ret;
}

void GenerateProductEqualConstraints(const ::pir::Value& lhs_tensor,
                                     const ::pir::Value& rhs_tensor,
                                     DimFunctions* ret) {
  List<ShapeDialectTensorDim> lhs_tensor_dims =
      MakeShapeDialectTensorDimList(lhs_tensor);
  List<ShapeDialectTensorDim> rhs_tensor_dims =
      MakeShapeDialectTensorDimList(rhs_tensor);

  List<ShapeDialectTempDim> lhs_reciprocal_dims =
      GenerateReciprocalConstraints(lhs_tensor_dims, ret);
  List<ShapeDialectTempDim> rhs_reciprocal_dims =
      GenerateReciprocalConstraints(rhs_tensor_dims, ret);
  CHECK_EQ(lhs_reciprocal_dims->size(), lhs_tensor_dims->size());
  CHECK_EQ(rhs_reciprocal_dims->size(), rhs_tensor_dims->size());
  for (std::size_t i = 0; i < lhs_tensor_dims->size(); ++i) {
    const List<DimVar> in_dim_vars =
        CollectProductDimVarExceptIdx(rhs_tensor_dims, lhs_reciprocal_dims, i);
    (*ret)->emplace_back(
        DimProduct<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>{
            lhs_tensor_dims->at(i), in_dim_vars});
  }
  for (std::size_t i = 0; i < rhs_tensor_dims->size(); ++i) {
    const List<DimVar> in_dim_vars =
        CollectProductDimVarExceptIdx(lhs_tensor_dims, rhs_reciprocal_dims, i);
    (*ret)->emplace_back(
        DimProduct<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>{
            rhs_tensor_dims->at(i), in_dim_vars});
  }
}

std::vector<::pir::shape::SymbolicDimOp> CreateSymbolicDimsFromValue(
    const ::pir::Value& tensor,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  CHECK_NOTNULL(shape_analysis.get());
  std::vector<::pir::shape::SymbolicDimOp> dims =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(tensor);
  CHECK_EQ(dims.size(),
           hlir::framework::pir::CompatibleInfo::ValueShape(tensor).size());
  return dims;
}

std::string ToTxtString(const ShapeDialectTensorDim& tensor_dim) {
  std::string ret{};
  ret += hlir::framework::pir::CompatibleInfo::ValueName(tensor_dim.tensor) +
         "[" + std::to_string(tensor_dim.axis) + "]";
  return ret;
}

std::string ToTxtStringImpl(const ShapeDialectTensorDim& dim) {
  return ToTxtString(dim);
}

std::string ToTxtStringImpl(const ShapeDialectTempDim& dim) {
  return std::string("temp_") + std::to_string(dim.value().unique_id());
}

std::string ToTxtString(const DimVar& dim_var) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    dim_var.variant());
}

void GenerateDimEqualConstraints(
    const std::vector<::pir::shape::SymbolicDimOp>& lhs_dims,
    const std::vector<::pir::shape::SymbolicDimOp>& rhs_dims,
    const ::pir::Value& lhs_tensor,
    const ::pir::Value& rhs_tensor,
    const ::pir::SymbolicDimMgr* symbolic_dim_mgr,
    DimFunctions* ret) {
  VisitEachIdxPairOfTwoVectors(
      lhs_dims, rhs_dims, [&](std::size_t lhs_idx, std::size_t rhs_idx) {
        const ::pir::shape::SymbolicDimOp& lhs_dim = lhs_dims.at(lhs_idx);
        const ::pir::shape::SymbolicDimOp& rhs_dim = rhs_dims.at(rhs_idx);
        if (const_cast<::pir::SymbolicDimMgr*>(symbolic_dim_mgr)
                ->IsSymbolicDimEqual(lhs_dim, rhs_dim)) {
          ShapeDialectTensorDim lhs_adt_dim{lhs_tensor, lhs_idx};
          ShapeDialectTensorDim rhs_adt_dim{rhs_tensor, rhs_idx};
          VLOG(4) << "Dim Equal: " << ToTxtString(lhs_adt_dim)
                  << " == " << ToTxtString(rhs_adt_dim);
          (*ret)->emplace_back(DimIdentity<tOut<ShapeDialectTensorDim>,
                                           tIn<ShapeDialectTensorDim>>{
              lhs_adt_dim, rhs_adt_dim});
          (*ret)->emplace_back(DimIdentity<tOut<ShapeDialectTensorDim>,
                                           tIn<ShapeDialectTensorDim>>{
              rhs_adt_dim, lhs_adt_dim});
        }
      });
}

void BuildTensorShapeDialectConstraints(
    const ::pir::Value& lhs_tensor,
    const ::pir::Value& rhs_tensor,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis,
    DimFunctions* ret) {
  std::vector<::pir::shape::SymbolicDimOp> lhs_dims =
      CreateSymbolicDimsFromValue(lhs_tensor, shape_analysis);
  std::vector<::pir::shape::SymbolicDimOp> rhs_dims =
      CreateSymbolicDimsFromValue(lhs_tensor, shape_analysis);

  GenerateDimEqualConstraints(lhs_dims,
                              rhs_dims,
                              lhs_tensor,
                              rhs_tensor,
                              &shape_analysis->symbolicDimMgr(),
                              ret);

  if (shape_analysis->symbolicDimMgr().IsSymbolicDimProductEqual(
          ::pir::SymbolicDimProduct{lhs_dims},
          ::pir::SymbolicDimProduct{rhs_dims})) {
    GenerateProductEqualConstraints(lhs_tensor, rhs_tensor, ret);
  }
}

DimFunctions BuildGraphShapeDialectConstraints(
    const cinn::hlir::framework::pir::Group* group,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  DimFunctions ret{};
  for (const ::pir::Operation* op_node : group->ops) {
    VisitEachTensorPairOfOp(
        op_node, [&](const ::pir::Value& lhs, const ::pir::Value& rhs) {
          BuildTensorShapeDialectConstraints(lhs, rhs, shape_analysis, &ret);
        });
  }
  return ret;
}

}  // namespace

namespace {

std::unordered_set<std::string> GetAllOutputNames(
    const std::vector<::pir::Operation*>& nodes) {
  std::unordered_set<std::string> output_names;
  for (const auto* op_node : nodes) {
    for (const ::pir::Value& out_node :
         const_cast<::pir::Operation*>(op_node)->results()) {
      output_names.emplace(
          hlir::framework::pir::CompatibleInfo::ValueName(out_node));
    }
  }
  return output_names;
}

List<::pir::Value> GetFeedList(
    const std::vector<::pir::Operation*>& op_nodes,
    const std::unordered_set<std::string>& out_names) {
  List<::pir::Value> ret{};
  // if the op's input var name cannot found in out_names, it is the group's
  // feed var
  std::unordered_set<std::string> feed_names;
  for (const auto* op_node : op_nodes) {
    for (const ::pir::Value in_node : op_node->operands_source()) {
      const auto& node_id =
          hlir::framework::pir::CompatibleInfo::ValueName(in_node);
      if (!out_names.count(node_id) && !feed_names.count(node_id)) {
        feed_names.emplace(node_id);
        ret->emplace_back(in_node);
      }
    }
  }
  return ret;
}

template <typename DoEachT>
void VisitEachTensor(const List<::pir::Value>& tensors, const DoEachT& DoEach) {
  for (const auto& tensor : *tensors) {
    DoEach(tensor);
  }
}

::pir::shape::SymbolicDimOp GetSymbolicDimOp4TensorDim(
    const ShapeDialectTensorDim& tensor_dim,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  const auto& [tensor, axis] = tensor_dim;
  const auto& symbolic_dim_ops =
      shape_analysis->GetOrCreateSymbolicDimsForRankedValue(tensor);
  CHECK_LT(axis, symbolic_dim_ops.size());
  return symbolic_dim_ops.at(axis);
}

SymbolicDim GetOrNewSymbolicDim(
    const ShapeDialectTensorDim& target_tensor_dim,
    const std::unordered_map<ShapeDialectTensorDim, SymbolicDim>&
        tensor_dim2symbolic_Dim,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  const auto& target_symbolic_dim_op =
      GetSymbolicDimOp4TensorDim(target_tensor_dim, shape_analysis);
  for (const auto& [tensor_dim, symbolic_dim] : tensor_dim2symbolic_Dim) {
    const auto& cur_symblic_dim_op =
        GetSymbolicDimOp4TensorDim(tensor_dim, shape_analysis);
    if (shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
            target_symbolic_dim_op, cur_symblic_dim_op)) {
      return symbolic_dim;
    }
  }
  return SymbolicDim{UniqueId::New()};
}

std::unordered_map<DimVar, const DimExpr> MakeEquationStartExpr(
    const cinn::hlir::framework::pir::Group* group,
    const std::shared_ptr<::pir::ShapeConstraintIRAnalysis>& shape_analysis,
    std::unordered_map<SymbolicDim, ::pir::shape::SymbolicDimOp>*
        map_expr_symbolic2dialect_symbolic) {
  std::unordered_map<DimVar, const DimExpr> ret{};
  std::unordered_set<std::string> output_names = GetAllOutputNames(group->ops);
  List<::pir::Value> feed_tensors = GetFeedList(group->ops, output_names);
  std::unordered_map<ShapeDialectTensorDim, SymbolicDim>
      tensor_dim2symbolic_Dim{};
  VisitEachTensor(feed_tensors, [&](const ::pir::Value& tensor) {
    std::vector<int> shape =
        hlir::framework::pir::CompatibleInfo::ValueShape(tensor);
    for (std::size_t i = 0; i < shape.size(); ++i) {
      ShapeDialectTensorDim tensor_dim{tensor, i};
      if (shape.at(i) > 0) {
        CHECK(ret.emplace(tensor_dim, std::int64_t(shape.at(i))).second);
      } else if (shape.at(i) == -1) {
        SymbolicDim symbolic_dim = GetOrNewSymbolicDim(
            tensor_dim, tensor_dim2symbolic_Dim, shape_analysis);
        CHECK(ret.emplace(tensor_dim, symbolic_dim).second);
        CHECK(tensor_dim2symbolic_Dim.emplace(tensor_dim, symbolic_dim).second);
        map_expr_symbolic2dialect_symbolic->emplace(
            symbolic_dim,
            GetSymbolicDimOp4TensorDim(tensor_dim, shape_analysis));
      } else {
        LOG(FATAL) << "Dead code. Invalid tensor shape = " << shape.at(i);
      }
    }
  });
  return ret;
}

}  // namespace

namespace {

using DimGraphView = EquationGraphTopoWalker<DimVar, const DimFunction*>;

DimGraphView MakeEquationGraphView(const DimFunctions& dim_functions) {
  return Graph<DimVar, DimFunction>::New(dim_functions)->GetGraphView();
}

class DimIndexExprInferContext final {
 public:
  DimIndexExprInferContext(const DimIndexExprInferContext&) = delete;
  DimIndexExprInferContext(DimIndexExprInferContext&&) = delete;

  explicit DimIndexExprInferContext(
      const std::unordered_map<DimVar, const DimExpr>& dim_var2dim_expr)
      : dim_var2dim_expr_(dim_var2dim_expr) {}

  const DimExpr& GetValue(const DimVar& dim_var) const {
    CHECK(HasValue(dim_var));
    return dim_var2dim_expr_.at(dim_var);
  }

  auto SetValue(const DimVar& dim_var, const DimExpr& dim_expr) {
    return dim_var2dim_expr_.emplace(dim_var, dim_expr);
  }

  bool HasValue(const DimVar& dim_var) const {
    return dim_var2dim_expr_.count(dim_var) > 0;
  }

 private:
  std::unordered_map<DimVar, const DimExpr> dim_var2dim_expr_;
};

std::unordered_map<DimVar, DimExpr> InferValuesImpl(
    const DimReciprocal<tOut<ShapeDialectTempDim>, tIn<ShapeDialectTensorDim>>&
        dim_reciprocal,
    const DimIndexExprInferContext* ctx) {
  std::unordered_map<DimVar, DimExpr> ret{};
  const auto& [out_temp_dim, in_tensor_dim] = dim_reciprocal.tuple();
  ret.emplace(out_temp_dim.value(),
              Reciprocal<DimExpr>(ctx->GetValue(in_tensor_dim.value())));
  return ret;
}

std::unordered_map<DimVar, DimExpr> InferValuesImpl(
    const DimProduct<tOut<ShapeDialectTensorDim>, tIn<List<DimVar>>>&
        dim_product,
    const DimIndexExprInferContext* ctx) {
  std::unordered_map<DimVar, DimExpr> ret{};
  const auto& [out_tensor_dim, in_dim_vars] = dim_product.tuple();
  List<DimExpr> in_dim_exprs{};
  for (const auto& in_dim_var : *(in_dim_vars.value())) {
    in_dim_exprs->emplace_back(ctx->GetValue(in_dim_var));
  }
  ret.emplace(out_tensor_dim.value(), Product<DimExpr>{in_dim_exprs});
  return ret;
}

std::unordered_map<DimVar, DimExpr> InferValuesImpl(
    const DimIdentity<tOut<ShapeDialectTensorDim>, tIn<ShapeDialectTensorDim>>&
        dim_identity,
    const DimIndexExprInferContext* ctx) {
  std::unordered_map<DimVar, DimExpr> ret{};
  const auto& [out_tensor_dim, in_tensor_dim] = dim_identity.tuple();
  ret.emplace(out_tensor_dim.value(), ctx->GetValue(in_tensor_dim.value()));
  return ret;
}

std::unordered_map<DimVar, DimExpr> InferValues(
    const DimFunction* function, const DimIndexExprInferContext* ctx) {
  return std::visit(
      [&](const auto& impl) { return InferValuesImpl(impl, ctx); },
      function->variant());
}

void MergeInferedValuesIntoCtx(const DimFunction* function,
                               DimIndexExprInferContext* ctx) {
  auto output_variable2value = InferValues(function, ctx);
  for (const auto& [dim_var, unsimplified_value] : output_variable2value) {
    DimExpr simplified_dim_expr = SimplifyDimExpr(unsimplified_value);
    if (!ctx->HasValue(dim_var)) {
      ctx->SetValue(dim_var, simplified_dim_expr);
    } else {
      const DimExpr& old_dim_expr = ctx->GetValue(dim_var);
      if (simplified_dim_expr != old_dim_expr) {
        LOG(FATAL) << "DimExpr Conflict! old_dim_expr = "
                   << ToTxtString(old_dim_expr)
                   << ", new_dim_expr = " << ToTxtString(simplified_dim_expr);
      }
    }
  }
}

std::unordered_set<::pir::Value> CollectAllTensors(
    const cinn::hlir::framework::pir::Group* group) {
  std::unordered_set<::pir::Value> ret{};
  for (const ::pir::Operation* op : group->ops) {
    for (const ::pir::Value& tensor : op->operands_source()) {
      ret.emplace(tensor);
    }
    for (const ::pir::Value& tensor :
         const_cast<::pir::Operation*>(op)->results()) {
      ret.emplace(tensor);
    }
  }
  return ret;
}

std::unordered_map<::pir::Value, std::vector<std::optional<DimExpr>>>
MakeValue2DimExpr(const cinn::hlir::framework::pir::Group* group,
                  const DimIndexExprInferContext* ctx) {
  std::unordered_set<::pir::Value> tensors = CollectAllTensors(group);
  std::unordered_map<::pir::Value, std::vector<std::optional<DimExpr>>> ret{};

  for (const ::pir::Value& tensor : tensors) {
    int rank = hlir::framework::pir::CompatibleInfo::ValueShape(tensor).size();
    std::vector<std::optional<DimExpr>> dim_exprs{};
    dim_exprs.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      ShapeDialectTensorDim tensor_dim{tensor, i};
      if (ctx->HasValue(tensor_dim)) {
        dim_exprs.emplace_back(ctx->GetValue(tensor_dim));
      } else {
        dim_exprs.emplace_back(std::nullopt);
      }
    }
    CHECK(ret.emplace(tensor, dim_exprs).second);
  }
  return ret;
}

std::unordered_map<::pir::Value, std::vector<std::optional<DimExpr>>>
SolveShapeDialectConstraints(
    const cinn::hlir::framework::pir::Group* group,
    const DimFunctions& dim_functions,
    const std::unordered_map<DimVar, const DimExpr>& equation_start) {
  const DimGraphView& graph_view = MakeEquationGraphView(dim_functions);
  auto infer_ctx = std::make_shared<DimIndexExprInferContext>(equation_start);

  std::vector<DimVar> start_vars{};
  for (const auto& [dim_var, _] : equation_start) {
    start_vars.emplace_back(dim_var);
  }

  graph_view.WalkFunction(
      start_vars.begin(), start_vars.end(), [&](const DimFunction* function) {
        MergeInferedValuesIntoCtx(function, infer_ctx.get());
      });

  return MakeValue2DimExpr(group, infer_ctx.get());
}

}  // namespace

void GraphSymbolicDimInferCtx::InitTensorDimExpr() {
  if (!FLAGS_cinn_enable_map_expr_dynamic_shape) {
    return;
  }
  DimFunctions dim_functions =
      BuildGraphShapeDialectConstraints(group_, group_->shape_analysis);
  std::unordered_map<DimVar, const DimExpr> equation_start =
      MakeEquationStartExpr(
          group_, group_->shape_analysis, &map_expr_symbolic2dialect_symbolic_);

  tensor2dim_exprs_ =
      SolveShapeDialectConstraints(group_, dim_functions, equation_start);
}

}  // namespace cinn::adt::config
