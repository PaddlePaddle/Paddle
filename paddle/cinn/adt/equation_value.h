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

namespace cinn::adt::equation::value {

DEFINE_ADT_TAG(tSymbolic);
DEFINE_ADT_TAG(tPointer);
DEFINE_ADT_TAG(tIterator);

using IterVar = tIterator<UniqueId>;

// Undefined = {}
struct Undefined final {};

#define DEFINE_ADT_UNARY(name)    \
  template <typename T>           \
  struct name : public Tuple<T> { \
    using Tuple<T>::Tuple;        \
  }

#define DEFINE_ADT_BINARY(name)        \
  template <typename T0, typename T1>  \
  struct name : public Tuple<T0, T1> { \
    using Tuple<T0, T1>::Tuple;        \
  }

DEFINE_ADT_UNARY(Neg);
DEFINE_ADT_BINARY(Add);
DEFINE_ADT_BINARY(Mul);
DEFINE_ADT_BINARY(Div);
DEFINE_ADT_BINARY(Mod);

DEFINE_ADT_UNION(Constant,
                 std::size_t,
                 tSymbolic<UniqueId>,
                 Neg<Constant>,
                 Add<Constant, Constant>,
                 Mul<Constant, Constant>);

// IndexDot T = Dot [Constant] T
template <typename StrideT, typename T>
struct Dot : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexDot final : public Dot<Constant, T> {
  using Dot<Constant, T>::Dot;
};

// IndexUnDot T = UnDot [Constant] T
template <typename StrideT, typename T>
struct UnDot : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexUnDot final : public UnDot<Constant, T> {
  using UnDot<Constant, T>::UnDot;
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

}  // namespace cinn::adt::equation::value
