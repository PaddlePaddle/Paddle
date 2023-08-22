// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

namespace cinn {
namespace adt {
namespace m_expr {

class TensorFlattenIndexExprNode;
// TensorFlattenIndexExpr = Box TensorFlattenIndexExprNode
using TensorFlattenIndexExpr = Box<TensorFlattenIndexExprNode>;

class TensorFlattenIndexLogicalExprNode;
// TensorFlattenIndexLogicalExpr = Box TensorFlattenIndexLogicalExprNode
using TensorFlattenIndexLogicalExpr = Box<TensorFlattenIndexLogicalExprNode>;

#define DEFINE_BINARY_EXPR(name, T)       \
  class name final : public Tuple<T, T> { \
    using Tuple<T, T>::Tuple;             \
  };

DEFINE_BINARY_EXPR(Add, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(Sub, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(Mul, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(Div, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(Mod, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(LE, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(LT, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(GE, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(GT, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(EQ, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(NE, TensorFlattenIndexExpr);
DEFINE_BINARY_EXPR(LogicalAnd, TensorFlattenIndexLogicalExpr);
DEFINE_BINARY_EXPR(LogicalOr, TensorFlattenIndexLogicalExpr);

// Let [Var Name] [TensorFlattenIndexExpr] TensorFlattenIndexExpr
class Let final : public Tuple<List<tVar<Name>>,
                               List<TensorFlattenIndexExpr>,
                               TensorFlattenIndexExpr> {
  using Tuple<List<tVar<Name>>,
              List<TensorFlattenIndexExpr>,
              TensorFlattenIndexExpr>::Tuple;
};

// LogicalNot TensorFlattenIndexLogicalExpr
class LogicalNot final : public Tuple<TensorFlattenIndexLogicalExpr> {
  using Tuple<TensorFlattenIndexLogicalExpr>::Tuple;
}

// clang-format off
/*
TensorFlattenIndexExprNode = Var Name
                           | Int64
                           | Add TensorFlattenIndexExpr TensorFlattenIndexExpr
                           | Sub TensorFlattenIndexExpr TensorFlattenIndexExpr
                           | Mul TensorFlattenIndexExpr TensorFlattenIndexExpr
                           | Div TensorFlattenIndexExpr TensorFlattenIndexExpr
                           | Mod TensorFlattenIndexExpr TensorFlattenIndexExpr
                           | Let [Var Name] [TensorFlattenIndexExpr] TensorFlattenIndexExpr
*/
// clang-format on
class TensorFlattenIndexExprNode final
    : public Union<tVar<Name>, std::int64_t, Add, Sub, Mul, Div, Mod, Let> {
  using Union<tVar<Name>, std::int64_t, Add, Sub, Mul, Div, Mod, Let>::Union;
};

// clang-format off
/*
TensorFlattenIndexLogicalExprNode = LE TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | LT TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | GE TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | GT TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | EQ TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | NE TensorFlattenIndexExpr TensorFlattenIndexExpr
                                  | LogicalAnd TensorFlattenIndexLogicalExpr TensorFlattenIndexLogicalExpr
                                  | LogicalOr TensorFlattenIndexLogicalExpr TensorFlattenIndexLogicalExpr
                                  | LogicalNot TensorFlattenIndexLogicalExpr
*/
// clang-format on
class TensorFlattenIndexLogicalExprNode final
    : public Union<LE, LT, GE, GT, EQ, NE, LogicalAnd, LogicalOr, LogicalNot> {
  using Union<LE, LT, GE, GT, EQ, NE, LogicalAnd, LogicalOr, LogicalNot>::Union;
};

// TensorFlattenIndexAssert = (TensorFlattenIndexLogicalExpr, tAssertMsg
// $std::string)
using TensorFlattenIndexAssert =
    Tuple<TensorFlattenIndexLogicalExpr, tAssertMsg<std::string>>;

// TensorFlattenIndexLambda = ([Var Name], TensorFlattenIndexExpr, Optional
// TensorFlattenIndexAssert)
using TensorFlattenIndexLambda = Tuple<List<tVar<Name>>,
                                       TensorFlattenIndexExpr,
                                       Optional<TensorFlattenIndexAssert>>;

}  // namespace m_expr
}  // namespace adt
}  // namespace cinn
