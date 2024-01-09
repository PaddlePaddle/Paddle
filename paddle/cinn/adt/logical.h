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
#include "paddle/cinn/adt/tree.h"

namespace cinn::adt {

DEFINE_ADT_BINARY(EQ);
DEFINE_ADT_BINARY(LT);
DEFINE_ADT_BINARY(GT);
DEFINE_ADT_BINARY(NE);
DEFINE_ADT_BINARY(GE);
DEFINE_ADT_BINARY(LE);
DEFINE_ADT_BINARY(And);
DEFINE_ADT_BINARY(Or);
DEFINE_ADT_UNARY(Not);

template <typename ValueT>
DEFINE_ADT_UNION(Compare,
                 EQ<ValueT, ValueT>,
                 LT<ValueT, ValueT>,
                 GT<ValueT, ValueT>,
                 NE<ValueT, ValueT>,
                 GE<ValueT, ValueT>,
                 LE<ValueT, ValueT>);

template <typename T>
DEFINE_ADT_UNION(LogicalOp, And<T, T>, Or<T, T>, Not<T>);

template <typename ValueT>
using Logical = Tree<LogicalOp, Compare<ValueT>>;

}  // namespace cinn::adt
