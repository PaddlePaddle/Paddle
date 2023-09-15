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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

namespace cinn::hlir::framework {
class Node;
class NodeData;
}  // namespace cinn::hlir::framework

namespace cinn::adt::equation {

DEFINE_ADT_TAG(tPointer);
DEFINE_ADT_TAG(tIterVar);

using IterVar = tIterVar<UniqueId>;

OVERLOAD_OPERATOR_EQ_NE(IterVar, TagEqual);

DEFINE_ADT_UNION(Constant,
                 std::int64_t,
                 tStride<UniqueId>,
                 tDim<UniqueId>,
                 List<Constant>,
                 Neg<Constant>,
                 Add<Constant, Constant>,
                 Mul<Constant, Constant>);

OVERLOAD_OPERATOR_EQ_NE(Constant, UnionEqual);
OVERLOAD_OPERATOR_EQ_NE(tStride<UniqueId>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(tDim<UniqueId>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(List<Constant>, ListEqual);
OVERLOAD_OPERATOR_EQ_NE(Neg<Constant>, TupleEqual);
using AddConstant = Add<Constant, Constant>;
using MulConstant = Mul<Constant, Constant>;
OVERLOAD_OPERATOR_EQ_NE(AddConstant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(MulConstant, TupleEqual);

template <typename IteratorsT>
struct IndexDot : public Tuple<IteratorsT, Constant> {
  using Tuple<IteratorsT, Constant>::Tuple;

  const IteratorsT& GetIterators() const { return std::get<0>(this->tuple()); }
};

template <typename IteratorsT>
struct IndexUnDot : public Tuple<IteratorsT, Constant> {
  using Tuple<IteratorsT, Constant>::Tuple;

  const IteratorsT& GetIterators() const { return std::get<0>(this->tuple()); }
};

// ConstantAdd T = Add T Constant
template <typename T>
struct ConstantAdd final : public Add<T, Constant> {
  using Add<T, Constant>::Add;

  const T& GetArg0() const { return std::get<0>(this->tuple()); }
};

// ConstantDiv T = Div T Constant
template <typename T>
struct ConstantDiv final : public Div<T, Constant> {
  using Div<T, Constant>::Div;

  const T& GetArg0() const { return std::get<0>(this->tuple()); }
};

// ConstantMod T = Mod T Constant
template <typename T>
struct ConstantMod final : public Mod<T, Constant> {
  using Mod<T, Constant>::Mod;

  const T& GetArg0() const { return std::get<0>(this->tuple()); }
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

DEFINE_ADT_UNION(Value,
                 Undefined,
                 Ok,
                 LoopDescriptor,
                 List<Value>,
                 IndexDot<Value>,
                 IndexUnDot<Value>,
                 ConstantAdd<Value>,
                 ConstantDiv<Value>,
                 ConstantMod<Value>,
                 ListGetItem<Value, Constant>,
                 PtrGetItem<Value>);

OVERLOAD_OPERATOR_EQ_NE(Value, UnionEqual);
OVERLOAD_OPERATOR_EQ_NE(List<Value>, ListEqual);
OVERLOAD_OPERATOR_EQ_NE(IndexDot<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(IndexUnDot<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantAdd<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantDiv<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantMod<Value>, TupleEqual);
using ListGetItem_Value_Constant = ListGetItem<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(ListGetItem_Value_Constant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(PtrGetItem<Value>, TupleEqual);

}  // namespace cinn::adt::equation

namespace std {

template <>
struct hash<cinn::adt::equation::IterVar> final {
  std::size_t operator()(const cinn::adt::equation::IterVar& iter_var) const {
    return iter_var.value().unique_id();
  }
};

}  // namespace std
