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

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn::fusion {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;
static OpPatternKind GetOpPatternKind(const ::pir::Operation* op) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*op);
}

static size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

// FIXME(Aurelius84): 0D Tensor is not compitable with other rank.
// So we need to add a special case for 0D Tensor.
static size_t GetCompitableRank(pir::Value value) {
  size_t rank = GetRank(value);
  return rank == 0 ? 1 : rank;
}

static std::vector<int64_t> GetReduceAxisIdx(pir::Operation* reduce_op) {
  const size_t input_rank = GetCompitableRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("axis");
  CHECK(attr_val.isa<::pir::ArrayAttribute>());
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  if (axis_attr.empty()) {
    // dim: [] means reduce_all.
    std::vector<int64_t> all_axis;
    for (int i = 0; i < input_rank; ++i) {
      all_axis.push_back(i);
    }
    return all_axis;
  }
  std::vector<int64_t> reduce_axis_idx;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
    reduce_axis_idx.push_back(axis);
  }
  VLOG(4) << "GetReduceAxisIdx: " << utils::Join(reduce_axis_idx, ",");
  return reduce_axis_idx;
}

static bool GetReduceOpKeepDims(pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keepdim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>().data();
}

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(const_cast<pir::Operation*>(op));
    ss << "\n";
  }
  return ss.str();
}

static std::optional<std::pair<pir::Value, pir::Value>>
GetBroadcastOpInputOuputValue(pir::Operation* op) {
  auto* mut_op = const_cast<pir::Operation*>(op);
  if (op->isa<paddle::dialect::ExpandOp>()) {
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
    return std::make_pair(expand_op.x(), expand_op.out());
  } else if (op->isa<cinn::dialect::BroadcastOp>()) {
    auto broadcast_op = mut_op->dyn_cast<cinn::dialect::BroadcastOp>();
    return std::make_pair(broadcast_op.x(), broadcast_op.out());
  } else {
    CHECK(false) << "Unsupported broadcast op: " << op->name();
  }
  return std::nullopt;
}

template <typename T>
void RemoveFromVector(std::vector<T>* vec, T item) {
  auto iter = std::find(vec->begin(), vec->end(), item);
  if (iter != vec->end()) {
    vec->erase(iter);
  }
}

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& first,
                            const std::vector<T>& second) {
  std::vector<T> result = first;
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

template <typename T>
std::vector<T> ConcatAll(const std::vector<std::vector<T>>& all) {
  std::vector<T> result;
  for (const auto& vec : all) {
    result = ConcatVector(result, vec);
  }
  return result;
}

template <typename T, typename F>
std::vector<T> FilterVector(const std::vector<T>& first, const F& func) {
  std::vector<T> result;
  for (const auto& i : first) {
    if (func(i)) {
      result.push_back(i);
    }
  }
  return result;
}

template <class A, class B>
std::vector<B> MapVector(const std::vector<A>& as,
                         const std::function<B(A)>& func) {
  std::vector<B> res;
  for (const auto& a : as) {
    res.push_back(func(a));
  }
  return res;
}

template <typename T>
std::set<T> ToSet(const std::vector<T>& input) {
  std::set<T> result(input.begin(), input.end());
  return result;
}

template <typename T>
bool IsAnyFirstInSecond(const std::vector<T>& first,
                        const std::vector<T>& second) {
  const auto& second_set = ToSet(second);
  for (const auto& ele : first) {
    if (second_set.count(ele)) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::vector<T> UniqueVectorBySet(const std::vector<T>& v) {
  std::unordered_set<T> unique(v.begin(), v.end());
  return std::vector<T>(unique.begin(), unique.end());
}

template <typename T>
void ExtendVector(std::vector<T>* first, const std::vector<T>& second) {
  std::unordered_set<T> visited =
      std::unordered_set<T>(first->begin(), first->end());
  for (auto iter = second.begin(); iter != second.end(); iter++) {
    if (visited.find(*iter) == visited.end()) {
      visited.emplace(*iter);
      first->emplace_back(*iter);
    }
  }
}

template <typename T>
std::vector<T> UniqueConcatVector(const std::vector<T>& first,
                                  const std::vector<T>& second) {
  std::vector<T> result = std::vector<T>(first);
  ExtendVector(&result, second);
  return result;
}

struct ValueDim {
  pir::Value v_;
  size_t idx_ = -1;
  std::weak_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
  ValueDim(pir::Value v, size_t idx) : v_(v), idx_(idx) {
    // Just get a related op to get the shape analysis. It can be value's
    // upstream op (defining op) or downstream op (user op).
    const auto GetRelatedOpFromValue =
        [](const pir::Value& v) -> pir::Operation* {
      if (v.defining_op() != nullptr) {
        return v.defining_op();
      }
      // For inputs of the program, the defining_op is nullptr, we use it's user
      // as the related op.
      PADDLE_ENFORCE_EQ(v.use_empty(),
                        false,
                        phi::errors::PreconditionNotMet(
                            "Value is an input value, it should have a use."));
      return v.first_use().owner();
    };
    shape_analysis_ = pir::ShapeAnalysisManager::Instance()
                          .Get(GetRelatedOpFromValue(v)->GetParentProgram())
                          .shared_from_this();
  }
  ValueDim() = default;
  ValueDim(const ValueDim& v) = default;
  bool operator==(const ValueDim& v) const {
    return (idx_ == v.idx_) && (v_ == v.v_);
  }

  symbol::DimExpr GetSymbolicDim() const {
    PADDLE_ENFORCE_NOT_NULL(v_.impl(), "Empty value is not expected.");
    return shape_analysis().GetProductDimExpr(v_, {static_cast<int>(idx_)});
  }

  bool empty() const { return idx_ == -1; }

  bool SymbolicEqualTo(const ValueDim& other) const {
    return shape_analysis().IsEqual(GetSymbolicDim(), other.GetSymbolicDim());
  }

  std::string DebugStr() const {
    std::ostringstream oss;
    oss << "ValueDim: ";
    oss << "Index: " << idx_;
    oss << ", ";
    v_.defining_op()->Print(oss);
    return oss.str();
  }

  pir::ShapeConstraintIRAnalysis& shape_analysis() const {
    auto shape_analysis_ptr = shape_analysis_.lock();
    PADDLE_ENFORCE_NOT_NULL(
        shape_analysis_ptr,
        ::common::errors::PreconditionNotMet("shape_analysis_ptr is nullptr."));
    return *shape_analysis_ptr;
  }
};

static std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v) {
  std::vector<ValueDim> value_dims;
  size_t rank = GetCompitableRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dims.emplace_back(v, i);
  }
  return value_dims;
}

struct ValueDimHash {
  std::size_t operator()(const ValueDim& p) const {
    auto h1 = std::hash<size_t>{}(p.idx_);
    auto h2 = std::hash<pir::Value>{}(p.v_);
    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ (h2 << 1);
  }
};

static std::vector<symbol::DimExpr> GetDimExprsFromValue(pir::Value value) {
  const auto& value_dims = GetAllValueDimFromValue(value);
  VLOG(4) << "Start Print:";
  std::function<symbol::DimExpr(ValueDim)> func =
      [](const ValueDim& value_dim) {
        const auto& symbolic_dim = value_dim.GetSymbolicDim();
        VLOG(4) << symbolic_dim;
        return symbolic_dim;
      };
  return MapVector(value_dims, func);
}

template <typename T, typename Int>
std::vector<T> GatherVector(const std::vector<T>& inp,
                            std::vector<Int> gathers) {
  std::vector<T> result;
  for (auto i : gathers) {
    result.push_back(inp[i]);
  }
  return result;
}

template <typename Int>
std::vector<Int> ExcludeIndex(int n, std::vector<Int> excludes) {
  std::vector<Int> result;
  for (int i = 0; i < n; ++i) {
    if (std::find(excludes.begin(), excludes.end(), i) == excludes.end()) {
      result.push_back(i);
    }
  }
  return result;
}

template <typename T, typename U>
std::vector<T> GatherVectorExcept(const std::vector<T>& source,
                                  const std::vector<U>& idx) {
  std::vector<T> result;
  for (U i = 0; i < source.size(); i++) {
    if (std::find(idx.begin(), idx.end(), i) == idx.end()) {
      result.emplace_back(source[i]);
    }
  }
  return result;
}

template <typename T>
std::vector<T> SliceVector(const std::vector<T>& inp, int start, int end) {
  if (start < 0) {
    start = inp.size() + start;
  }
  if (end < 0) {
    end = inp.size() + end;
  }
  std::vector<T> result;
  for (int i = start; i < end; ++i) {
    result.push_back(inp[i]);
  }
  return result;
}

template <typename T, typename U>
std::vector<U> VectorFlatMap(
    const std::vector<T>& inp,
    const std::function<std::vector<U>(const T&)>& func) {
  std::vector<U> result;
  for (const auto& i : inp) {
    result = ConcatVector(result, func(i));
  }
  return result;
}

inline std::vector<pir::Value> GetInputsValue(
    const std::vector<pir::Operation*>& ops) {
  // include middle value.
  std::function<std::vector<pir::Value>(pir::Operation* const&)> get_inputs =
      [](const pir::Operation* const& in) { return in->operands_source(); };
  const auto& all_inputs =
      VectorFlatMap<pir::Operation*, pir::Value>(ops, get_inputs);
  return UniqueVectorBySet(all_inputs);
}

inline std::vector<pir::Value> GetOutputsValue(
    const std::vector<pir::Operation*>& ops) {
  // include middle value.
  std::function<std::vector<pir::Value>(pir::Operation* const&)> get_outputs =
      [](const pir::Operation* const& in) { return in->results(); };
  const auto& all_outputs =
      VectorFlatMap<pir::Operation*, pir::Value>(ops, get_outputs);
  return UniqueVectorBySet(all_outputs);
}

template <typename T>
std::vector<T> VectorDiff(const std::vector<T>& left,
                          const std::vector<T>& right) {
  const auto& set = ToSet(right);
  std::vector<T> res;
  for (const auto& v : left) {
    if (!set.count(v)) res.push_back(v);
  }
  return res;
}

inline bool All(const std::vector<bool> a) {
  bool res = true;
  for (bool i : a) {
    res &= i;
  }
  return res;
}

inline bool Any(const std::vector<bool> a) {
  bool res = false;
  for (bool i : a) {
    res |= i;
  }
  return res;
}

}  // namespace cinn::fusion
