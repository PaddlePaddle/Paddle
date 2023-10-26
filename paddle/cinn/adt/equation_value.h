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

#pragma once

#include <functional>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/match.h"

namespace cinn::hlir::framework {
class Node;
class NodeData;
}  // namespace cinn::hlir::framework

namespace cinn::adt {

DEFINE_ADT_TAG(tPointer);

template <typename IteratorsT, typename DimsT>
struct IndexDotValue : public Tuple<IteratorsT, DimsT> {
  using Tuple<IteratorsT, DimsT>::Tuple;

  const IteratorsT& GetIteratorsValue() const {
    return std::get<0>(this->tuple());
  }
};

template <typename IndexT, typename DimsT>
struct IndexUnDotValue : public Tuple<IndexT, DimsT> {
  using Tuple<IndexT, DimsT>::Tuple;

  const IndexT& GetIndexValue() const { return std::get<0>(this->tuple()); }
};

// ListGetItem T ConstantT = (T, ConstantT)
template <typename T, typename ConstantT>
struct ListGetItem final : public Tuple<T, ConstantT> {
  using Tuple<T, ConstantT>::Tuple;

  const T& GetList() const { return std::get<0>(this->tuple()); }
};

// PtrGetItem T = (tPointer UniqueId, T)
template <typename T>
struct PtrGetItem final : public Tuple<tPointer<UniqueId>, T> {
  using Tuple<tPointer<UniqueId>, T>::Tuple;

  const T& GetArg1() const { return std::get<1>(this->tuple()); }
};

template <typename ValueT, typename ConstantT>
struct BroadcastedIterator final : public Tuple<ValueT, ConstantT> {
  using Tuple<ValueT, ConstantT>::Tuple;

  const ValueT& GetArg0() const { return std::get<0>(this->tuple()); }
};

DEFINE_ADT_UNION(Value,
                 Undefined,
                 Ok,
                 Iterator,
                 Constant,
                 List<Value>,
                 IndexDotValue<Value, Constant>,
                 IndexUnDotValue<Value, Constant>,
                 ListGetItem<Value, Constant>,
                 BroadcastedIterator<Value, Constant>,
                 PtrGetItem<Value>);

OVERLOAD_OPERATOR_EQ_NE(Value, UnionEqual);
using IndexDot_Value_Constant = IndexDotValue<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(IndexDot_Value_Constant, TupleEqual);
using IndexUnDot_Value_Constant = IndexUnDotValue<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(IndexUnDot_Value_Constant, TupleEqual);
using ListGetItem_Value_Constant = ListGetItem<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(ListGetItem_Value_Constant, TupleEqual);
using BroadcastedIterator_Value_Constant = BroadcastedIterator<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(BroadcastedIterator_Value_Constant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(PtrGetItem<Value>, TupleEqual);

inline std::size_t GetHashValue(const Value& value);

inline std::size_t GetHashValueImpl(const Undefined& value) { return 0; }
inline std::size_t GetHashValueImpl(const Ok& value) { return 1; }
inline std::size_t GetHashValueImpl(const Iterator& value) {
  return value.value().unique_id();
}
inline std::size_t GetHashValueImpl(const Constant& value) {
  return GetHashValue(value);
}
inline std::size_t GetHashValueImpl(const List<Value>& value) {
  std::size_t ret = 0;
  for (const auto& v : *value) {
    ret = hash_combine(ret, GetHashValue(v));
  }
  return ret;
}
inline std::size_t GetHashValueImpl(
    const IndexDotValue<Value, Constant>& value) {
  const auto& [v, c] = value.tuple();
  return hash_combine(GetHashValue(v), GetHashValue(c));
}
inline std::size_t GetHashValueImpl(
    const IndexUnDotValue<Value, Constant>& value) {
  const auto& [v, c] = value.tuple();
  return hash_combine(GetHashValue(v), GetHashValue(c));
}
inline std::size_t GetHashValueImpl(const ListGetItem<Value, Constant>& value) {
  const auto& [v, c] = value.tuple();
  return hash_combine(GetHashValue(v), GetHashValue(c));
}
inline std::size_t GetHashValueImpl(
    const BroadcastedIterator<Value, Constant>& value) {
  const auto& [v, c] = value.tuple();
  return hash_combine(GetHashValue(v), GetHashValue(c));
}
inline std::size_t GetHashValueImpl(const PtrGetItem<Value>& value) {
  const auto& [pointer, c] = value.tuple();
  return hash_combine(pointer.value().unique_id(), GetHashValue(c));
}

OVERRIDE_UNION_GET_HASH_VALUE(Value);

}  // namespace cinn::adt

namespace std {

template <>
struct hash<cinn::adt::Value> {
  std::size_t operator()(const cinn::adt::Value& value) const {
    return GetHashValue(value);
  }
};

}  // namespace std
