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
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

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
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::ArrayAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The axis attribute should be an array."));
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
    PADDLE_ENFORCE_GE(
        axis,
        0,
        ::common::errors::InvalidArgument(
            "The 'axis' must be greater than or equal to 0, but received %d.",
            axis));

    PADDLE_ENFORCE_LT(axis,
                      input_rank,
                      ::common::errors::InvalidArgument(
                          "The 'axis' must be less than 'input_rank', but "
                          "received axis = %d and input_rank = %d.",
                          axis,
                          input_rank));

    reduce_axis_idx.push_back(axis);
  }
  VLOG(4) << "GetReduceAxisIdx: " << utils::Join(reduce_axis_idx, ",");
  return reduce_axis_idx;
}

static bool GetReduceOpKeepDims(pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keepdim");
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::BoolAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The keepdim attribute should be a bool."));
  return attr_val.dyn_cast<::pir::BoolAttribute>().data();
}

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    pir::Operation* op);

static std::vector<std::pair<size_t, size_t>> GetNonBroadCastDims(
    pir::Operation* op) {
  std::vector<std::pair<size_t, size_t>> res;
  auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  CHECK(broad_cast_value.has_value());

  const auto& [input_value, output_value] = broad_cast_value.value();
  const int input_rank = GetRank(input_value);
  const int output_rank = GetRank(output_value);
  CHECK_GE(output_rank, input_rank);

  // Compare axis one by one, from back to front.
  // The rule of broadcasting:
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_cn.html#id7
  for (int i = 1; i <= input_rank; ++i) {
    int input_axis = input_rank - i;
    int output_axis = output_rank - i;
    if (input_axis < 0 || output_axis < 0) break;
    if (shape_analysis->IsProductEqual(
            input_value, {input_axis}, output_value, {output_axis})) {
      res.emplace_back(input_axis, output_axis);
    }
  }

  return res;
}

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(const_cast<pir::Operation*>(op));
    ss << "(" << op << ")"
       << "\n";
  }
  return ss.str();
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
std::unordered_set<T> ToUnorderedSet(const std::vector<T>& input) {
  std::unordered_set<T> result(input.begin(), input.end());
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
                        ::common::errors::PreconditionNotMet(
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

template <typename T>
bool AnyTargetInCandidate(const std::vector<T>& targets,
                          const std::vector<T>& candidate) {
  std::unordered_set<T> pool = ToUnorderedSet(candidate);
  for (const auto& item : targets) {
    if (pool.find(item) != pool.end()) {
      return true;
    }
  }
  return false;
}

static std::vector<pir::Operation*> FindDownstreamOps(pir::Operation* op) {
  std::vector<pir::Operation*> result;
  for (int i = 0; i < op->num_results(); i++) {
    auto v = op->result(i);
    for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
         ++consumer_it) {
      result.emplace_back(consumer_it->owner());
    }
  }
  return result;
}

static const size_t GetUsageIdx(const pir::Value& v, pir::Operation* op) {
  size_t i = 0;
  for (auto consumer_it = v.use_begin(); consumer_it != v.use_end();
       ++consumer_it, ++i) {
    if (consumer_it->owner() == op) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the usage of value %s in op %s", v.impl(), op->name()));
}

static const size_t GetOperandIdx(const pir::Value& v, pir::Operation* op) {
  for (size_t i = 0; i < op->num_operands(); i++) {
    if (op->operand(i).source() == v) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the value %s as operand of op %s", v.impl(), op->name()));
}

static const size_t GetResultIdx(const pir::Value& v, pir::Operation* op) {
  size_t i = 0;
  for (size_t i = 0; i < op->num_results(); i++) {
    if (op->result(i) == v) {
      return i;
    }
  }
  PADDLE_THROW(::common::errors::NotFound(
      "Can not find the value %s as result of op %s", v.impl(), op->name()));
}

static bool IsDirectUpstream(const pir::Operation* upstream,
                             const pir::Operation* downstream) {
  for (const auto& value : downstream->results()) {
    for (const auto& operand : upstream->operands()) {
      if (value == operand.source()) {
        return true;
      }
    }
  }
  return false;
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
