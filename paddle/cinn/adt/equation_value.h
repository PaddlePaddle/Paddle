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

#include "cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/match_and_rewrite.h"

namespace cinn::adt::equation {

DEFINE_ADT_TAG(tPointer);
DEFINE_ADT_TAG(tIterVar);

using IterVar = tIterVar<UniqueId>;

OVERLOAD_OPERATOR_EQ_NE(IterVar, TagEqual);

DEFINE_ADT_UNION(Constant,
                 std::int64_t,
                 tStride<UniqueId>,
                 tDim<UniqueId>,
                 Neg<Constant>,
                 Add<Constant, Constant>,
                 Mul<Constant, Constant>);

OVERLOAD_OPERATOR_EQ_NE(Constant, UnionEqual);
OVERLOAD_OPERATOR_EQ_NE(tStride<UniqueId>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(tDim<UniqueId>, TagEqual);
OVERLOAD_OPERATOR_EQ_NE(Neg<Constant>, TupleEqual);
using AddConstant = Add<Constant, Constant>;
using MulConstant = Mul<Constant, Constant>;
OVERLOAD_OPERATOR_EQ_NE(AddConstant, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(MulConstant, TupleEqual);

// Undefined = {}
struct Undefined final {
  bool operator==(const Undefined&) const { return true; }
  bool operator!=(const Undefined&) const { return false; }
};

// IndexDot T = DotValue [Constant] T
template <typename StrideT, typename T>
struct DotValue : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexDot final : public DotValue<Constant, T> {
  using DotValue<Constant, T>::DotValue;
};

// IndexUnDot T = UnDotValue [Constant] T
template <typename StrideT, typename T>
struct UnDotValue : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexUnDot final : public UnDotValue<Constant, T> {
  using UnDotValue<Constant, T>::UnDotValue;
};

// ConstantAdd T = Add T Constant
template <typename T>
struct ConstantAdd final : public Add<T, Constant> {
  using Add<T, Constant>::Add;
};

// ConstantDiv T = Div T Constant
template <typename T>
struct ConstantDiv final : public Div<T, Constant> {
  using Div<T, Constant>::Div;
};

// ConstantMod T = Mod T Constant
template <typename T>
struct ConstantMod final : public Mod<T, Constant> {
  using Mod<T, Constant>::Mod;
};

// ListGetItem T = (T, Constant)
template <typename T>
struct ListGetItem final : public Tuple<T, Constant> {
  using Tuple<T, Constant>::Tuple;
};

// PtrGetItem T = (tPointer UniqueId, T)
template <typename T>
struct PtrGetItem final : public Tuple<tPointer<UniqueId>, T> {
  using Tuple<tPointer<UniqueId>, T>::Tuple;
};

DEFINE_ADT_UNION(Value,
                 Undefined,
                 IterVar,
                 List<Value>,
                 IndexDot<Value>,
                 IndexUnDot<Value>,
                 ConstantAdd<Value>,
                 ConstantDiv<Value>,
                 ConstantMod<Value>,
                 ListGetItem<Value>,
                 PtrGetItem<Value>);

OVERLOAD_OPERATOR_EQ_NE(Value, UnionEqual);
OVERLOAD_OPERATOR_EQ_NE(List<Value>, ListEqual);
OVERLOAD_OPERATOR_EQ_NE(IndexDot<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(IndexUnDot<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantAdd<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantDiv<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ConstantMod<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(ListGetItem<Value>, TupleEqual);
OVERLOAD_OPERATOR_EQ_NE(PtrGetItem<Value>, TupleEqual);

}  // namespace cinn::adt::equation
