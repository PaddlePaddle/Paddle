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

namespace cinn::adt {

DEFINE_ADT_UNARY(Negative);
template <typename T>
using Neg = Negative<T>;
DEFINE_ADT_UNARY(Reciprocal);
DEFINE_ADT_BINARY(Add);
DEFINE_ADT_BINARY(Sub);
DEFINE_ADT_BINARY(Mul);
DEFINE_ADT_BINARY(Div);
DEFINE_ADT_BINARY(Mod);

template <typename T>
struct Sum final {
  List<T> operands;

  const Sum& tuple() const { return *this; }
};

template <typename T>
struct Product final {
  List<T> operands;

  const Product& tuple() const { return *this; }
};

// Arithmetic T = Neg T
//              | Add T T
//              | Sub T T
//              | Mul T T
//              | Div T T
//              | Mod T T
template <typename ValueT>
DEFINE_ADT_UNION(Arithmetic,
                 Neg<ValueT>,
                 Add<ValueT, ValueT>,
                 Sub<ValueT, ValueT>,
                 Mul<ValueT, ValueT>,
                 Div<ValueT, ValueT>,
                 Mod<ValueT, ValueT>);
}  // namespace cinn::adt
