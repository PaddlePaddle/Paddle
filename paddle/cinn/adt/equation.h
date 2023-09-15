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
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"

namespace cinn::adt {

class UniqueId final {
 public:
  UniqueId() : unique_id_(NewSeqNumber()) {}
  UniqueId(const UniqueId&) = default;
  UniqueId(UniqueId&&) = default;
  UniqueId& operator=(const UniqueId&) = default;
  UniqueId& operator=(UniqueId&&) = default;

  static UniqueId New() { return UniqueId{NewSeqNumber()}; }

  bool operator==(const UniqueId& other) const {
    return this->unique_id_ == other.unique_id_;
  }

  bool operator!=(const UniqueId& other) const {
    return !this->operator==(other);
  }

  bool operator<(const UniqueId& other) const {
    return this->unique_id_ < other.unique_id_;
  }

  std::size_t unique_id() const { return unique_id_; }

 private:
  static std::size_t NewSeqNumber() {
    static std::atomic<std::size_t> seq_number{0};
    return ++seq_number;
  }

  explicit UniqueId(std::size_t unique_id) : unique_id_(unique_id) {}
  std::size_t unique_id_;
};

}  // namespace cinn::adt

namespace std {

template <>
struct hash<cinn::adt::UniqueId> final {
  std::size_t operator()(const cinn::adt::UniqueId& unique_id) const {
    return unique_id.unique_id();
  }
};

}  // namespace std

namespace cinn::adt {

// Iterator = tIterator UniqueId
using Iterator = tIterator<UniqueId>;
// IteratorTuple = [Iterator]
using IteratorTuple = List<Iterator>;
// Index = tIndex UniqueId
using Index = tIndex<UniqueId>;
// Dim = tDim UniqueId
using Dim = tDim<UniqueId>;
// DimTuple = [Dim]
using DimTuple = List<Dim>;
// Stride = tStride UniqueId
using Stride = tStride<UniqueId>;
// StrideTuple = [Stride]
using StrideTuple = List<Stride>;
// FakeOpPlaceHolder = tOpPlaceHolder UniqueId
using FakeOpPlaceHolder = tOpPlaceHolder<UniqueId>;

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

template <typename StrideT, typename OutT, typename InT>
struct Dot;

// Dot [Stride] (tOut Index) (tIn [Iterator])
template <>
struct Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>
    : public Tuple<List<Stride>, tOut<Index>, tIn<List<Iterator>>> {
  using Tuple<List<Stride>, tOut<Index>, tIn<List<Iterator>>>::Tuple;
};

template <typename StrideT, typename OutT, typename InT>
struct UnDot;

// UnDot [Stride] (tOut [Iterator]) (tIn Index)
template <>
struct UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>
    : public Tuple<List<Stride>, tOut<List<Iterator>>, tIn<Index>> {
  using Tuple<List<Stride>, tOut<List<Iterator>>, tIn<Index>>::Tuple;
};

// OpArgIndexes = (tIn [Index], tOut [Index])
struct OpArgIndexes final : public Tuple<tIn<List<Index>>, tOut<List<Index>>> {
  using Tuple<tIn<List<Index>>, tOut<List<Index>>>::Tuple;
};

template <typename OutT, typename InT>
struct InMsgBox2OutMsgBox;

// InMsgBox2OutMsgBox (tOut (tOutMsgBox OpArgIndexes)) (tIn (tInMsgBox
// OpArgIndexes))
template <>
struct InMsgBox2OutMsgBox<tOut<tOutMsgBox<OpArgIndexes>>,
                          tIn<tInMsgBox<OpArgIndexes>>>
    : public Tuple<FakeOpPlaceHolder,
                   tOut<tOutMsgBox<OpArgIndexes>>,
                   tIn<tInMsgBox<OpArgIndexes>>> {
  using Tuple<FakeOpPlaceHolder,
              tOut<tOutMsgBox<OpArgIndexes>>,
              tIn<tInMsgBox<OpArgIndexes>>>::Tuple;
};

// clang-format off
DEFINE_ADT_UNION(Equation,
                 Identity<tOut<Iterator>, tIn<Iterator>>,
                 Identity<tOut<Index>, tIn<Index>>,
                 Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>,
                 UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>,
                 InMsgBox2OutMsgBox<tOut<tOutMsgBox<OpArgIndexes>>,
                                    tIn<tInMsgBox<OpArgIndexes>>>);

// Variable = Iterator | Index | FakeOpPlaceHolder
DEFINE_ADT_UNION(Variable,
                 Iterator,
                 Index,
                 FakeOpPlaceHolder);
// clang-format on

OVERLOAD_OPERATOR_EQ_NE(Variable, UnionEqual);

// Function = Equation
using Function = Equation;

using Equations = List<Equation>;
using GraphView = EquationGraphTopoWalker<Variable, const Equation*>;

}  // namespace cinn::adt

namespace std {

template <>
struct hash<cinn::adt::Iterator> final {
  std::size_t operator()(const cinn::adt::Iterator& iterator) const {
    return iterator.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::Index> final {
  std::size_t operator()(const cinn::adt::Index& index) const {
    return index.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::FakeOpPlaceHolder> final {
  std::size_t operator()(
      const cinn::adt::FakeOpPlaceHolder& placeholder) const {
    return placeholder.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::Variable> final {
  std::size_t operator()(const cinn::adt::Variable& variable) const {
    std::size_t hash_value =
        variable >>
        cinn::adt::match{
            [](const cinn::adt::Iterator& iterator) {
              return std::hash<cinn::adt::Iterator>()(iterator);
            },
            [](const cinn::adt::Index& index) {
              return std::hash<cinn::adt::Index>()(index);
            },
            [](const cinn::adt::FakeOpPlaceHolder& fake_op_placeholder) {
              return std::hash<cinn::adt::FakeOpPlaceHolder>()(
                  fake_op_placeholder);
            }};
    return cinn::adt::hash_combine(hash_value, variable.variant().index());
  }
};

}  // namespace std
