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

#include <atomic>
#include <functional>
#include <string>
#include <type_traits>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/equation_variable.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"

namespace cinn::adt {

template <typename OutT, typename InT>
struct Identity;

// Identity (tOut Iterator) (tIn Iterator)
template <>
struct Identity<tOut<Iterator>, tIn<Iterator>>
    : public Tuple<tOut<Iterator>, tIn<Iterator>> {
  using Tuple<tOut<Iterator>, tIn<Iterator>>::Tuple;
};

// Identity (tOut Index) (tIn Index)
template <>
struct Identity<tOut<Index>, tIn<Index>>
    : public Tuple<tOut<Index>, tIn<Index>> {
  using Tuple<tOut<Index>, tIn<Index>>::Tuple;
};

template <typename DimT, typename OutT, typename InT>
struct IndexDot;

// IndexDot [DimExpr] (tOut Index) (tIn [Iterator])
template <>
struct IndexDot<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>
    : public Tuple<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>> {
  using Tuple<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>::Tuple;
};

template <typename DimT, typename OutT, typename InT>
struct IndexUnDot;

// IndexUnDot [DimExpr] (tOut [Iterator]) (tIn Index)
template <>
struct IndexUnDot<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>>
    : public Tuple<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>> {
  using Tuple<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>>::Tuple;
};

// OpArgIndexes = (tIn [Index], tOut [Index])
template <typename OutIndexT>
struct OpArgIndexes final
    : public Tuple<tIn<List<Index>>, tOut<List<OutIndexT>>> {
  using Tuple<tIn<List<Index>>, tOut<List<OutIndexT>>>::Tuple;
};

template <typename FakeOpT, typename OutT, typename InT>
struct InMsg2OutMsg;

// InMsg2OutMsg (tOut FakeOpPlaceHolder) (tOut (tOutMsg OpArgIndexes))
// (tIn (tInMsg OpArgIndexes))
template <>
struct InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                    tOut<OpArgIndexes<std::optional<Index>>>,
                    tIn<OpArgIndexes<Index>>>
    : public Tuple<tOut<FakeOpPlaceHolder>,
                   tOut<OpArgIndexes<std::optional<Index>>>,
                   tIn<OpArgIndexes<Index>>> {
  using Tuple<tOut<FakeOpPlaceHolder>,
              tOut<OpArgIndexes<std::optional<Index>>>,
              tIn<OpArgIndexes<Index>>>::Tuple;
};

template <typename T0, typename T1>
struct ConstantFunction;

template <>
struct ConstantFunction<tOut<Iterator>, tIn<Index>> final
    : public Tuple<tOut<Iterator>, tIn<Index>, DimExpr> {
  using Tuple<tOut<Iterator>, tIn<Index>, DimExpr>::Tuple;
};

template <typename DimT, typename OutT, typename InT>
struct GetBroadcastedIterator;

template <>
struct GetBroadcastedIterator<DimExpr, tOut<Iterator>, tIn<Iterator>>
    : public Tuple<DimExpr, tOut<Iterator>, tIn<Iterator>> {
  using Tuple<DimExpr, tOut<Iterator>, tIn<Iterator>>::Tuple;
};

// clang-format off
DEFINE_ADT_UNION(Equation,
                 Identity<tOut<Iterator>, tIn<Iterator>>,
                 Identity<tOut<Index>, tIn<Index>>,
                 GetBroadcastedIterator<DimExpr,
                                        tOut<Iterator>, tIn<Iterator>>,
                 IndexDot<List<DimExpr>, tOut<Index>,
                          tIn<List<Iterator>>>,
                 IndexUnDot<List<DimExpr>,
                            tOut<List<Iterator>>, tIn<Index>>,
                 InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                    tOut<OpArgIndexes<std::optional<Index>>>,
                                    tIn<OpArgIndexes<Index>>>,
                 ConstantFunction<tOut<Iterator>, tIn<Index>>);
// clang-format on

// Function = Equation
using Function = Equation;

using Equations = List<Equation>;
using GraphView = EquationGraphTopoWalker<Variable, const Equation*>;

std::string GetFunctionTypeName(const Function& function);

const void* GetFunctionDataPtr(const Function& function);

}  // namespace cinn::adt
