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

namespace cinn::hlir::framework {
class Node;
class NodeData;
}  // namespace cinn::hlir::framework

namespace cinn::adt {

DEFINE_ADT_TAG(tPointer);

template <typename IteratorsT, typename StridesT>
struct IndexDotValue : public Tuple<IteratorsT, StridesT> {
  using Tuple<IteratorsT, StridesT>::Tuple;

  const IteratorsT& GetIteratorsValue() const {
    return std::get<0>(this->tuple());
  }
};

template <typename IndexT, typename StridesT>
struct IndexUnDotValue : public Tuple<IndexT, StridesT> {
  using Tuple<IndexT, StridesT>::Tuple;

  const IndexT& GetIndexValue() const { return std::get<0>(this->tuple()); }
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
                 Iterator,
                 Constant,
                 List<Value>,
                 IndexDotValue<Value, Constant>,
                 IndexUnDotValue<Value, Constant>,
                 ConstantAdd<Value>,
                 ConstantDiv<Value>,
                 ConstantMod<Value>,
                 ListGetItem<Value, Constant>,
                 PtrGetItem<Value>);

OVERLOAD_OPERATOR_EQ_NE(Value, UnionEqual);
using IndexDot_Value_Constant = IndexDotValue<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(IndexDot_Value_Constant, TupleEqual);
using IndexUnDot_Value_Constant = IndexUnDotValue<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(IndexUnDot_Value_Constant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantAdd<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantDiv<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantMod<Value>, TupleEqual);
using ListGetItem_Value_Constant = ListGetItem<Value, Constant>;
OVERLOAD_OPERATOR_EQ_NE(ListGetItem_Value_Constant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(PtrGetItem<Value>, TupleEqual);

std::string DebugString(const Constant& c);
std::string DebugString(const Value& value);

}  // namespace cinn::adt
