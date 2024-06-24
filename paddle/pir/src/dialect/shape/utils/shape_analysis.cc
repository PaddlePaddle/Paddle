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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include <string>
#include "paddle/common/bfs_walker.h"
#include "paddle/common/topo_walker.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/src/core/value_impl.h"

namespace pir {

static std::string GetValueId(Value val) {
  auto op_id = val.defining_op()->id();
  auto val_idx = val.dyn_cast<OpResult>().index();

  return val.defining_op()->name() + "_" + std::to_string(op_id) + "_rst_" +
         std::to_string(val_idx);
}

void InferSymbolicShapeContext::Init() {
  value_id_to_shape_or_data_.clear();
  next_sym_idx_ = sym_idx_begin_;
  constraints_manager_.SetEqualCallbackFunc(
      [&](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
        return SubstituteDimExpr(lhs, rhs);
      });
}

void InferSymbolicShapeContext::RegisterSymbolConstraintFromContext(
    const InferSymbolicShapeContext& other) {
  PADDLE_ENFORCE_EQ(
      next_sym_idx_,
      0,
      common::errors::PreconditionNotMet("next_sym_idx_ should be 0 when init "
                                         "symbol constraint, but now get %d",
                                         next_sym_idx_));
  PADDLE_ENFORCE_EQ(value_id_to_shape_or_data_.size(),
                    0,
                    common::errors::PreconditionNotMet(
                        "value_id_to_shape_or_data_ should be empty when init "
                        "symbol constraint, but now get %d",
                        value_id_to_shape_or_data_.size()));
  sym_idx_begin_ = other.next_sym_idx_;
  next_sym_idx_ = sym_idx_begin_;
  // init equal constraints
  for (const auto& kv : other.constraints_manager_.equals().GetMap()) {
    constraints_manager_.AddEqCstr(kv.first, kv.second);
  }
  // init broadcastable constraints
  for (const auto& bc_item : other.constraints_manager_.broadcastables()) {
    constraints_manager_.AddBroadcastableCstr(bc_item.data->lhs,
                                              bc_item.data->rhs);
  }
  // init gtone constraints
  for (const auto& gt_one : other.constraints_manager_.gtones()) {
    constraints_manager_.AddGTOneCstr(gt_one);
  }

  substitution_pattern_ = other.substitution_pattern_;

  // TODO(Hongqing-work): change this func name and pybind after add backward
  // cache mechanism
  for (const auto& kv : other.infer_symbolic_shape_cache_) {
    infer_symbolic_shape_cache_[kv.first] = kv.second;
  }
}

const std::string InferSymbolicShapeContext::GetNextSymName() {
  return "S" + std::to_string(next_sym_idx_++);
}

bool InferSymbolicShapeContext::HasShapeOrDataForValue(Value val) const {
  if (!val) {
    return false;
  }
  return value_id_to_shape_or_data_.count(val.impl()->id()) > 0;
}

const symbol::ShapeOrDataDimExprs&
InferSymbolicShapeContext::GetShapeOrDataForValue(Value val) const {
  if (!val || !val.type()) {
    static auto null_shape_or_data =
        symbol::ShapeOrDataDimExprs(symbol::NullShapeOrDataDimExpr());
    return null_shape_or_data;
  }
  if (!HasShapeOrDataForValue(val)) {
    PADDLE_THROW(phi::errors::Fatal(
        "Fail to GetShapeOrDataForValue on InferSymbolicShape!"));
  }

  return value_id_to_shape_or_data_.at(val.impl()->id());
}

void InferSymbolicShapeContext::SetSymbolForValueByStaticShape(Value val) {
  const auto& value_type = val.type();
  if (!val || !value_type) {
    LOG(WARNING) << "Risk on SetSymbolForValueByStaticShape for null value";
    return;
  }
  if (!IsStaticShape(val)) {
    LOG(WARNING)
        << "Risk on SetSymbolForValueByStaticShape for contain_unknown_dim";
  }
  const auto& GetStaticShapeForDenseTensorType =
      [&](DenseTensorType type_info) -> symbol::TensorShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> static_shape;
    for (int i = 0; i < type_info.dims().size(); ++i) {
      int dim = type_info.dims()[i];
      if (dim > 0) {
        static_shape.emplace_back(dim);
      } else {
        static_shape.emplace_back(GetNextSymName());
      }
    }
    return symbol::TensorShapeOrDataDimExprs(static_shape);
  };

  if (value_type.isa<DenseTensorType>()) {
    const DenseTensorType& type_info = value_type.dyn_cast<DenseTensorType>();
    SetShapeOrDataForValue(val, GetStaticShapeForDenseTensorType(type_info));
    return;
  }
  if (value_type.isa<VectorType>()) {
    const std::vector<Type>& vec_data =
        value_type.dyn_cast<VectorType>().data();
    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    for (const auto& vec : vec_data) {
      if (!vec.isa<DenseTensorType>()) {
        PADDLE_THROW(phi::errors::Fatal(
            "Set static shape ONLY SUPPORT inner type DenseTensorType!"));
      } else {
        const DenseTensorType& type_info = vec.dyn_cast<DenseTensorType>();
        shape_data_list.emplace_back(
            GetStaticShapeForDenseTensorType(type_info));
      }
    }
    SetShapeOrDataForValue(val, shape_data_list);
    return;
  }
  PADDLE_THROW(phi::errors::Fatal(
      "Set static shape ONLY SUPPORT DenseTensorType and VectorType!"));
}

void InferSymbolicShapeContext::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  const symbol::ShapeOrDataDimExprs& simplified_shape_or_data =
      SimplifyBroadcastForShapeOrData(shape_or_data);
  const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
      symbol::SubstituteShapeOrData(simplified_shape_or_data,
                                    substitution_pattern_);
  if (!val) {
    LOG(WARNING) << "Set shape or data for null value";
    return;
  }
  auto iter = value_id_to_shape_or_data_.find(val.impl()->id());
  if (iter == value_id_to_shape_or_data_.end()) {
    value_id_to_shape_or_data_.emplace(val.impl()->id(),
                                       substituted_shape_or_data);
  } else {
    iter->second = substituted_shape_or_data;
  }
}

void InferSymbolicShapeContext::AddEqualCstr(const symbol::DimExpr& lhs,
                                             const symbol::DimExpr& rhs) {
  constraints_manager_.AddEqCstr(lhs, rhs);
}

bool InferSymbolicShapeContext::IsEqual(const symbol::DimExpr& lhs,
                                        const symbol::DimExpr& rhs) const {
  return constraints_manager_.IsEqual(lhs, rhs);
}

void InferSymbolicShapeContext::AddGreatThanOneCstr(
    const symbol::DimExpr& dim_expr) {
  constraints_manager_.AddGTOneCstr(dim_expr);
}

bool InferSymbolicShapeContext::IsGreatThanOne(
    const symbol::DimExpr& dim_expr) const {
  return constraints_manager_.IsGTOne(dim_expr);
}

void InferSymbolicShapeContext::AddBroadcastableCstr(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
  constraints_manager_.AddBroadcastableCstr(lhs, rhs);
}

bool InferSymbolicShapeContext::IsBroadcastable(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const {
  return constraints_manager_.IsBroadcastable(lhs, rhs);
}

symbol::ShapeOrDataDimExprs
InferSymbolicShapeContext::SimplifyBroadcastForShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto SimplifyBroadcast =
      [&](const symbol::Broadcast<symbol::DimExpr>& bc) -> symbol::DimExpr {
    const symbol::List<symbol::DimExpr>& dim_exprs = bc.operands;
    symbol::List<symbol::DimExpr> gtone_list;
    for (const auto& dim_expr : *dim_exprs) {
      if (IsGreatThanOne(dim_expr)) gtone_list->push_back(dim_expr);
    }
    symbol::DimExpr simplified_dim_expr = bc;
    if (gtone_list->size() == 1) {
      simplified_dim_expr = gtone_list->at(0);
    } else if (gtone_list->size() > 1) {
      for (size_t i = 1; i < gtone_list->size(); i++) {
        AddEqualCstr(gtone_list->at(0), gtone_list->at(i));
      }
      simplified_dim_expr = gtone_list->at(0);
    }
    return simplified_dim_expr;
  };

  auto DimExprsVisitor =
      [&](const std::vector<symbol::DimExpr>& original_dim_expr)
      -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> simplified_dim_exprs{};
    for (const symbol::DimExpr& dim_expr : original_dim_expr) {
      // TODO(jiawenxuan): recursively evaluate each dim expr
      if (dim_expr.isa<symbol::Broadcast<symbol::DimExpr>>()) {
        const auto& simplified_dim_expr = SimplifyBroadcast(
            dim_expr.Get<symbol::Broadcast<symbol::DimExpr>>());
        simplified_dim_exprs.push_back(simplified_dim_expr);
      } else {
        simplified_dim_exprs.push_back(dim_expr);
      }
    }
    return simplified_dim_exprs;
  };

  auto TensorShapeOrDataVisitor =
      [&](const symbol::TensorShapeOrDataDimExprs& shape_or_data)
      -> symbol::TensorShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> simplified_shape =
        DimExprsVisitor(shape_or_data.shape());
    if (!shape_or_data.data().has_value()) {
      return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape);
    } else {
      std::vector<symbol::DimExpr> simplified_data =
          DimExprsVisitor(shape_or_data.data().value());
      return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape,
                                                  simplified_data);
    }
  };

  return shape_or_data.Match(
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(
            TensorShapeOrDataVisitor(tensor_shape_or_data));
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs simplified_tensor_list;
        for (const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data :
             tensor_list) {
          simplified_tensor_list.push_back(
              TensorShapeOrDataVisitor(tensor_shape_or_data));
        }
        return symbol::ShapeOrDataDimExprs(simplified_tensor_list);
      },
      [&](const symbol::NullShapeOrDataDimExpr& null_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(null_shape_or_data);
      });
}

namespace {

bool CanSubstituteInShapeAnalysis(const symbol::DimExpr& lhs,
                                  const symbol::DimExpr& rhs) {
  auto CanSubstitutePredictor = ::common::Overloaded{
      [](std::int64_t lhs, const auto& rhs) { return true; },
      [](const std::string& lhs, const std::string& rhs) { return true; },
      [](const std::string& lhs,
         const symbol::Broadcast<symbol::DimExpr>& rhs) { return true; },
      [](const auto& lhs, const auto& rhs) { return false; }};
  return std::visit(CanSubstitutePredictor, lhs.variant(), rhs.variant()) ||
         std::visit(CanSubstitutePredictor, rhs.variant(), lhs.variant());
}

}  // namespace

void InferSymbolicShapeContext::SubstituteDimExpr(
    const symbol::DimExpr& origin, const symbol::DimExpr& substituted) {
  if (!CanSubstituteInShapeAnalysis(origin, substituted)) return;

  substitution_pattern_[origin] = substituted;
  for (auto& val : substitution_pattern_) {
    if (val.second == origin) {
      val.second = substituted;
    }
  }

  for (auto& val : value_id_to_shape_or_data_) {
    const symbol::ShapeOrDataDimExprs& substituted_shape_or_data =
        symbol::SubstituteShapeOrData(val.second, substitution_pattern_);
    val.second = substituted_shape_or_data;
  }

  decltype(infer_symbolic_shape_cache_) new_op_shape_share_cache;
  for (auto& item : infer_symbolic_shape_cache_) {
    std::vector<symbol::ShapeOrDataDimExprs> input_shape_or_datas =
        item.first.GetInputShapeOrDatas();
    std::vector<symbol::ShapeOrDataDimExprs> input_substituted_result;
    for (const auto& shape_or_data : input_shape_or_datas) {
      const auto& substituted_shape_or_data =
          symbol::SubstituteShapeOrData(shape_or_data, substitution_pattern_);
      input_substituted_result.emplace_back(substituted_shape_or_data);
    }
    pir::InferSymbolicShapeCacheKey new_infer_symbolic_shape_cache_key =
        item.first;
    new_infer_symbolic_shape_cache_key.SetInputShapeOrDatas(
        input_substituted_result);
    std::vector<symbol::ShapeOrDataDimExprs> output_substituted_result;
    for (const auto& shape_or_data : item.second) {
      const auto& substituted_shape_or_data =
          symbol::SubstituteShapeOrData(shape_or_data, substitution_pattern_);
      output_substituted_result.emplace_back(substituted_shape_or_data);
    }
    new_op_shape_share_cache[new_infer_symbolic_shape_cache_key] =
        output_substituted_result;
  }
  infer_symbolic_shape_cache_ = std::move(new_op_shape_share_cache);
}

void InferSymbolicShapeContext::PrintShapeOrDatas() const {
  LOG(INFO) << "shape analysis : @" << this
            << " value_id_to_shape_or_data_ size : "
            << value_id_to_shape_or_data_.size();
  LOG(INFO) << "----------- ShapeOrData for Values ------------";
  for (const auto& [value_id, shape_or_data] : value_id_to_shape_or_data_) {
    LOG(INFO) << value_id << " : " << shape_or_data;
  }
}

void InferSymbolicShapeContext::SetOpInferSymbolicShapeCache(
    const InferSymbolicShapeCacheKey& op_infer_cache_key,
    InferSymbolicShapeCacheValue result_shape) {
  infer_symbolic_shape_cache_[op_infer_cache_key] = result_shape;
}

std::optional<InferSymbolicShapeCacheValue>
InferSymbolicShapeContext::GetOpInferSymbolicShapeCache(
    const InferSymbolicShapeCacheKey& op_infer_cache_key) const {
  if (infer_symbolic_shape_cache_.count(op_infer_cache_key) != 0) {
    return infer_symbolic_shape_cache_.at(op_infer_cache_key);
  }
  return std::nullopt;
}

void ShapeConstraintIRAnalysis::Init() { context_.Init(); }

void ShapeConstraintIRAnalysis::RegisterSymbolConstraintFromShapeAnalysis(
    const ShapeConstraintIRAnalysis& other) {
  context_.RegisterSymbolConstraintFromContext(other.context_);
}

const std::string ShapeConstraintIRAnalysis::GetNextSymName() {
  return context_.GetNextSymName();
}

void ShapeConstraintIRAnalysis::SetSymbolForValueByStaticShape(Value val) {
  context_.SetSymbolForValueByStaticShape(val);
}

void ShapeConstraintIRAnalysis::InferShapeOrDataForValue(Value val) {
  std::unordered_set<Operation*> subgraph_ops;
  std::vector<Operation*> start_ops;
  const auto& GetRealOperandSource = [&](Operation* op) -> std::vector<Value> {
    if (op->num_regions() == 0) {
      return op->operands_source();
    } else {
      std::vector<Value> ret;
      for (uint32_t i = 0; i < op->num_regions(); i++) {
        for (auto& block : op->region(i)) {
          for (auto& sub_op : block) {
            for (auto& operand : sub_op.operands_source()) {
              ret.emplace_back(operand);
            }
          }
        }
      }
      return ret;
    }
  };

  const auto& VisitNotInferedInputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (auto& operand : GetRealOperandSource(op)) {
          if (operand.impl() && !context_.HasShapeOrDataForValue(operand)) {
            if (!operand.defining_op()) {
              SetSymbolForValueByStaticShape(operand);
            } else {
              Visit(operand.defining_op());
            }
          }
        }
      };

  ::common::BfsWalker<Operation*> build_subgraph_walker(VisitNotInferedInputOp);
  build_subgraph_walker(val.defining_op(), [&](Operation* op) {
    subgraph_ops.insert(op);
    bool has_prev_op = false;
    for (auto& operand : GetRealOperandSource(op)) {
      if (operand.impl() && !context_.HasShapeOrDataForValue(operand)) {
        if (!operand.defining_op()) {
          SetSymbolForValueByStaticShape(operand);
        } else {
          has_prev_op = true;
        }
      }
    }
    if (!has_prev_op) {
      start_ops.emplace_back(op);
    }
  });

  const auto& VisitSubgraphInputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (auto& operand : GetRealOperandSource(op)) {
          if (operand.impl() && subgraph_ops.count(operand.defining_op())) {
            Visit(operand.defining_op());
          }
        }
      };
  const auto& VisitSubgraphOutputOp =
      [&](Operation* op, const std::function<void(Operation*)>& Visit) {
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          for (auto iter = op->result(i).use_begin();
               iter != op->result(i).use_end();
               ++iter) {
            auto parent_op = iter->owner();
            while (parent_op) {
              if (subgraph_ops.count(parent_op)) {
                Visit(parent_op);
                break;
              }
              parent_op = parent_op->GetParentOp();
            }
          }
        }
      };
  ::common::TopoWalker<Operation*> topo_infer_walker(VisitSubgraphInputOp,
                                                     VisitSubgraphOutputOp);

  topo_infer_walker(start_ops.begin(), start_ops.end(), [&](Operation* op) {
    auto infer_symbolic_shape_interface =
        op->dyn_cast<pir::InferSymbolicShapeInterface>();
    if (infer_symbolic_shape_interface) {
      infer_symbolic_shape_interface.InferSymbolicShape(&context_);
      for (auto& result_value : op->results()) {
        if (!result_value || !result_value.type()) {
          continue;
        }
        if (!context_.HasShapeOrDataForValue(result_value)) {
          PADDLE_THROW(phi::errors::Fatal(op->name() +
                                          " HAS ERROR on InferSymbolicShape!"));
        }
      }
    } else {
      LOG(WARNING) << op->name()
                   << " DOES NOT have InferSymbolicShapeInterface!";
      for (auto& result_value : op->results()) {
        if (!result_value || !result_value.type()) {
          continue;
        }
        if (!context_.HasShapeOrDataForValue(result_value)) {
          SetSymbolForValueByStaticShape(result_value);
        }
      }
    }
  });
}

const symbol::ShapeOrDataDimExprs&
ShapeConstraintIRAnalysis::GetShapeOrDataForValue(Value val) {
  if (!val || !val.type()) {
    static auto null_shape_or_data =
        symbol::ShapeOrDataDimExprs(symbol::NullShapeOrDataDimExpr());
    return null_shape_or_data;
  }
  if (!context_.HasShapeOrDataForValue(val)) {
    // backtrack to infer shape from defining op
    if (!val.defining_op()) {
      SetSymbolForValueByStaticShape(val);
    } else {
      VLOG(3) << "InferShapeOrDataForValue,  defining_op: "
              << val.defining_op()->name();
      InferShapeOrDataForValue(val);
    }
  }

  return context_.GetShapeOrDataForValue(val);
}

void ShapeConstraintIRAnalysis::SetShapeOrDataForValue(
    Value val, const symbol::ShapeOrDataDimExprs& shape_or_data) {
  context_.SetShapeOrDataForValue(val, shape_or_data);
}

bool ShapeConstraintIRAnalysis::IsEqual(const symbol::DimExpr& lhs,
                                        const symbol::DimExpr& rhs) const {
  return context_.IsEqual(lhs, rhs);
}

bool ShapeConstraintIRAnalysis::IsGreatThanOne(
    const symbol::DimExpr& dim_expr) const {
  return context_.IsGreatThanOne(dim_expr);
}

bool ShapeConstraintIRAnalysis::IsBroadcastable(
    const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) const {
  return context_.IsBroadcastable(lhs, rhs);
}

void ShapeConstraintIRAnalysis::PrintShapeOrDatas() const {
  context_.PrintShapeOrDatas();
}

// Currently, we only support TensorShapeOrDataDimExprs but not
// TensorListShapeOrDataDimExprs to compare the shape.
bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  PADDLE_ENFORCE_EQ(
      lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
          rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
      true,
      phi::errors::InvalidArgument(
          "Currently, IsShapeEqual only support TensorShapeOrDataDimExprs "
          "but not TensorListShapeOrDataDimExprs."));

  // For static shape, directly compare the shapes.
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    return lhs_type.GetShape() == rhs_type.GetShape();
  }

  // For dynamic shape, compare the symbolic dimensions.
  return lhs_shape_data.variant() == rhs_shape_data.variant();
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs,
    const std::vector<int>& lhs_dim_idxs,
    Value rhs,
    const std::vector<int>& rhs_dim_idxs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    int64_t lhs_product = 1;
    int64_t rhs_product = 1;
    for (int i : lhs_dim_idxs) {
      lhs_product *= lhs_type.GetShape()[i];
    }
    for (int i : rhs_dim_idxs) {
      rhs_product *= rhs_type.GetShape()[i];
    }
    return lhs_product == rhs_product;
  }

  auto lhs_shape_data = GetShapeOrDataForValue(lhs);
  auto rhs_shape_data = GetShapeOrDataForValue(rhs);

  PADDLE_ENFORCE_EQ(
      lhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>() &&
          rhs_shape_data.isa<symbol::TensorShapeOrDataDimExprs>(),
      true,
      phi::errors::InvalidArgument(
          "Currently, IsProductEqual only support TensorShapeOrDataDimExprs "
          "but not TensorListShapeOrDataDimExprs."));

  symbol::DimExpr lhs_product(1);
  symbol::DimExpr rhs_product(1);
  if (!lhs_shape_data.shape().empty()) {
    for (int i : lhs_dim_idxs) {
      lhs_product = lhs_product * lhs_shape_data.shape()[i];
    }
  }
  if (!rhs_shape_data.shape().empty()) {
    for (int i : rhs_dim_idxs) {
      rhs_product = rhs_product * rhs_shape_data.shape()[i];
    }
  }
  return symbol::SimplifyDimExpr(lhs_product) ==
         symbol::SimplifyDimExpr(rhs_product);
}

bool ShapeConstraintIRAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

bool ShapeConstraintIRAnalysis::IsSameNumel(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  // For static shape
  if (lhs_type.IsStaticShape() && rhs_type.IsStaticShape()) {
    auto lhs_shape = lhs_type.GetShape();
    auto rhs_shape = rhs_type.GetShape();
    if (lhs_shape == rhs_shape) {
      return true;
    }
    return common::product(lhs_shape) == common::product(rhs_shape);
  }

  return IsProductEqual(lhs,
                        0,
                        static_cast<int>(lhs_type.GetRank()),
                        rhs,
                        0,
                        static_cast<int>(rhs_type.GetRank()));
}

symbol::DimExpr ShapeConstraintIRAnalysis::GetProductDimExpr(
    Value value, const std::vector<int>& dim_idxs) {
  // For static shape
  auto value_type = value.type().dyn_cast<ShapedTypeInterface>();
  if (value_type.IsStaticShape()) {
    int64_t product = 1;
    for (int i : dim_idxs) {
      product *= value_type.GetShape()[i];
    }
    return symbol::DimExpr{product};
  }

  // For dynamic shape
  const auto& shape_data = GetShapeOrDataForValue(value);
  symbol::DimExpr product{1};
  for (int i : dim_idxs) {
    product = product * shape_data.shape()[i];
  }
  return symbol::SimplifyDimExpr(product);
}

pir::PrintHooks ShapeConstraintIRAnalysis::PrintHook() {
  pir::PrintHooks print_hook;
  print_hook.op_print_hook = [&](Operation* op, IrPrinter& printer) {
    printer.IrPrinter::PrintOperation(op);
    printer.os << " { ";
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      if (context_.HasShapeOrDataForValue(op->result(i))) {
        printer.os << "(" << this->GetShapeOrDataForValue(op->result(i)) << ")";
      } else {
        printer.os << "()";
      }
      if (i < op->num_results() - 1) {
        printer.os << ", ";
      }
    }
    printer.os << " }";
    printer.os << "\t(op_" << op->id() << ")";
  };
  return print_hook;
}

ShapeAnalysisManager& ShapeAnalysisManager::Instance() {
  static ShapeAnalysisManager instance;
  return instance;
}

ShapeConstraintIRAnalysis& ShapeAnalysisManager::Get(
    const pir::Program* program) {
  auto it = tables_.find(program->module_op().operation()->id());

  if (it == tables_.end()) {
    it = tables_
             .emplace(program->module_op().operation()->id(),
                      std::make_shared<ShapeConstraintIRAnalysis>())
             .first;
  }

  return *it->second;
}

bool IsStaticShape(const Value& value) {
  const auto& value_type = value.type();
  if (!value || !value_type) {
    return false;
  }
  if (value_type.isa<DenseTensorType>()) {
    return !::common::contain_unknown_dim(
        value_type.dyn_cast<DenseTensorType>().dims());
  }
  if (value_type.isa<VectorType>()) {
    bool is_static = true;
    auto vec_data = value_type.dyn_cast<VectorType>().data();
    for (const auto& vec : vec_data) {
      if (!vec.isa<DenseTensorType>()) {
        is_static = false;
        break;
      } else {
        is_static = !::common::contain_unknown_dim(
            vec.dyn_cast<DenseTensorType>().dims());
        if (!is_static) {
          break;
        }
      }
    }
    return is_static;
  }
  return false;
}

static const char* kOpCallStack = "op_callstack";
static const char* kSymShapeStr = "sym_shape_str";
static const char* kResultName = "name";

InferSymbolicShapeCacheKey::InferSymbolicShapeCacheKey(
    const Operation& op,
    const std::vector<symbol::ShapeOrDataDimExprs>& input_shape_or_datas)
    : InferSymbolicShapeCacheKey(
          op.name(), input_shape_or_datas, op.attributes()) {}

InferSymbolicShapeCacheKey::InferSymbolicShapeCacheKey(
    const std::string& op_name,
    const std::vector<symbol::ShapeOrDataDimExprs>& input_shape_or_datas,
    const AttributeMap& attributes)
    : op_name_(op_name), input_shape_or_datas_(input_shape_or_datas) {
  // Keep attribute always in order.
  std::map<std::string, ::pir::Attribute, std::less<>> order_attributes(
      attributes.begin(), attributes.end());
  attributes_.reserve(attributes.size());
  for (const auto& [attr_name, attr_value] : order_attributes) {
    if (!attr_value || attr_name == kOpCallStack || attr_name == kSymShapeStr ||
        attr_name == kResultName)
      continue;
    attributes_.emplace_back(attr_name, attr_value);
  }
}

std::size_t InferSymbolicShapeCacheKey::GetHashValue() const {
  const auto name_hash_func = std::hash<std::string>();
  const auto attr_hash_func = std::hash<pir::Attribute>();
  const auto shape_hash_func = std::hash<symbol::ShapeOrDataDimExprs>();
  std::size_t res = name_hash_func(op_name_);
  for (const auto& item : attributes_) {
    res = pir::detail::hash_combine(res, name_hash_func(item.first));
    res = pir::detail::hash_combine(res, attr_hash_func(item.second));
  }
  for (const auto& item : input_shape_or_datas_) {
    res = pir::detail::hash_combine(res, shape_hash_func(item));
  }
  return res;
}

bool InferSymbolicShapeCacheKey::operator==(
    const InferSymbolicShapeCacheKey& other) const {
  if (op_name_ != other.op_name_) return false;
  if (attributes_.size() != other.attributes_.size()) return false;
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    if (attributes_[i].first != other.attributes_[i].first ||
        attributes_[i].second != other.attributes_[i].second)
      return false;
  }
  if (input_shape_or_datas_.size() != other.input_shape_or_datas_.size())
    return false;
  for (std::size_t i = 0; i < input_shape_or_datas_.size(); ++i) {
    if (input_shape_or_datas_[i] != other.input_shape_or_datas_[i])
      return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os,
                         const InferSymbolicShapeCacheKey& info) {
  os << "InferSymbolicShapeCacheKey - " << info.op_name_ << std::endl;
  if (!info.attributes_.empty()) {
    os << "  attrs: {";
    for (std::size_t i = 0; i < info.attributes_.size() - 1; ++i) {
      ::pir::IrPrinter(os).PrintAttribute(info.attributes_[i].second);
      os << ", ";
    }
    ::pir::IrPrinter(os).PrintAttribute(info.attributes_.back().second);
    os << std::endl;
  }
  if (!info.input_shape_or_datas_.empty()) {
    os << "  input_shape_or_datas: {";
    for (std::size_t i = 0; i < info.input_shape_or_datas_.size() - 1; ++i) {
      os << info.input_shape_or_datas_[i] << ", ";
    }
    os << info.input_shape_or_datas_.back() << "}" << std::endl;
  }
  return os;
}

const std::vector<symbol::ShapeOrDataDimExprs>&
InferSymbolicShapeCacheKey::GetInputShapeOrDatas() const {
  return input_shape_or_datas_;
}
void InferSymbolicShapeCacheKey::SetInputShapeOrDatas(
    const std::vector<symbol::ShapeOrDataDimExprs>& input_shape_or_datas) {
  this->input_shape_or_datas_ = input_shape_or_datas;
}

}  // namespace pir
