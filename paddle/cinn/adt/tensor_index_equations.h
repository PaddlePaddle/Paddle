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

#include <string>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/m_ir.h"

namespace cinn {
namespace adt {
namespace m_ir {

class TensorIndexExprNode;
// TensorIndexExpr = Box TensorIndexExprNode
using TensorIndexExpr = Box<TensorIndexExprNode>;

class Add final : public Tuple<TensorIndexExpr, TensorIndexExpr> {
  using Tuple<TensorIndexExpr, TensorIndexExpr>::Tuple;
};

class Sub final : public Tuple<TensorIndexExpr, TensorIndexExpr> {
  using Tuple<TensorIndexExpr, TensorIndexExpr>::Tuple;
};

class TensorIndexExprNode final
    : public Union<std::int64_t, tIndexVar<Name>, Add, Sub> {
  using Union<std::int64_t, tIndexVar<Name>, Add, Sub>::Union;
};

// DimSize T = Int64 | T
template <typename T>
class DimSize final : public Union<std::int64_t, T> {
  using Union<std::int64_t, T>::Union;
};

// Product T
template <typename T>
class Product final {};

// Stride T = Product T
template <typename T>
class Stride final : public Product<T> {
  using Product<T>::Product;
};

// FlattenNdIndexDot VarNameT DimT = ([VarNameT], [Stride DimT])
template <typename VarNameT, typename DimT>
class FlattenNdIndexDot final
    : public Tuple<List<VarNameT>, List<Stride<DimT>>> {
  using Tuple<List<VarNameT>, List<Stride<DimT>>>::Tuple;
};

// FlattenTensorIndexDot T = FlattenNdIndexDot (tIndexVar Name) (DimSize T)
template <typename T>
class FlattenTensorIndexDot final
    : public FlattenNdIndexDot<tIndexVar<Name>, DimSize<T>> {
  using FlattenNdIndexDot<tIndexVar<Name>, DimSize<T>>::FlattenNdIndexDot;
};

// FlattenScheduleIndexDot = FlattenNdIndexDot (tScheduleIterVar Name)
// ScheduleSize
using FlattenScheduleIndexDot =
    FlattenNdIndexDot<tScheduleIterVar<Name>, ScheduleSize>;

// Equal T = (T, T)
// Equal T0 T1 = (T0, T1)
template <typename T0, typename T1 = T0>
class Equal final : public Tuple<T0, T1> {
  using Tuple<T0, T1>::Tuple;
};

// clang-format off
/*
TensorIndexEquation T = Equal TensorIndexExpr TensorIndexExpr
                      | Equal (FlattenTensorIndexDot T) (FlattenTensorIndexDot T)
                      | Equal (FlattenTensorIndexDot T) (tTensorSize Name)
                      | Equal (tTensorSize Name) FlattenScheduleIndexDot
*/
// clang-format on
template <typename T>
class TensorIndexEquation final
    : Union<Equal<TensorIndexExpr>,
            Equal<FlattenTensorIndexDot<T>>,
            Equal<FlattenTensorIndexDot<T>, tTensorSize<Name>>,
            Equal<tTensorSize<Name>, FlattenScheduleIndexDot>> {
  using Union<Equal<TensorIndexExpr>,
              Equal<FlattenTensorIndexDot<T>>,
              Equal<FlattenTensorIndexDot<T>, tTensorSize<Name>>,
              Equal<tTensorSize<Name>, FlattenScheduleIndexDot>>::Union;
};

// T = SymbolicDim or FactorSymbolicDim
// TensorIndexEquations T = [TensorIndexEquation T]
template <typename T>
class TensorIndexEquations final : public List<TensorIndexEquation<T>> {
  using List<TensorIndexEquation<T>>::List;
};

}  // namespace m_ir
}  // namespace adt
}  // namespace cinn
