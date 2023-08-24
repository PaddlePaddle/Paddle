// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

namespace cinn::adt::equation {

class UniqueId final {
 public:
  UniqueId(const UniqueId&) = default;
  UniqueId(UniqueId&&) = default;

  static UniqueId New() {
    static std::atomic<std::uint64_t> seq_number{0};
    return UniqueId{++seq_number};
  }

  bool operator==(const UniqueId& other) const {
    return this->unique_id_ == other.unique_id_;
  }

  bool operator!=(const UniqueId& other) const {
    return !this->operator==(other);
  }

  bool operator<(const UniqueId& other) const {
    return this->unique_id_ < other.unique_id_;
  }

  std::uint64_t unique_id() const { return unique_id_; }

 private:
  explicit UniqueId(std::uint64_t unique_id) : unique_id_(unique_id) {}
  std::uint64_t unique_id_;
};

}  // namespace cinn::adt::equation

namespace std {

template <>
struct hash<cinn::adt::equation::UniqueId> final {
  std::size_t operator()(const cinn::adt::equation::UniqueId& unique_id) const {
    return unique_id.unique_id();
  }
};

}  // namespace std

namespace cinn::adt::equation {

DEFINE_ADT_TAG(tIterator);
DEFINE_ADT_TAG(tIndex);
DEFINE_ADT_TAG(tDim);
DEFINE_ADT_TAG(tStride);
DEFINE_ADT_TAG(tOp);

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
// FakeOpPlaceHolder = tOp Name
using FakeOpPlaceHolder = tOp<Name>;

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

template <typename OutT, typename InT>
struct ConstructFakeOpPlaceHolder;

// ConstructFakeOpPlaceHolder (tOut FakeOpPlaceHolder) (tIn [Index])
template <>
struct ConstructFakeOpPlaceHolder<tOut<FakeOpPlaceHolder>, tIn<List<Index>>>
    : public Tuple<tOut<FakeOpPlaceHolder>, tIn<List<Index>>> {
  using Tuple<tOut<FakeOpPlaceHolder>, tIn<List<Index>>>::Tuple;
};

// clang-format off
/*
Equation = Identity (tOut Iterator) (tIn Iterator)
         | Identity (tOut Index) (tIn Index)
         | Dot [Stride] (tOut Index) (tIn [Iterator])
         | ConstructFakeOpPlaceHolder (tOut FakeOpPlaceHolder) (tIn [Index])
*/
DEFINE_ADT_UNION(Equation,
                 Identity<tOut<Iterator>, tIn<Iterator>>,
                 Identity<tOut<Index>, tIn<Index>>,
                 Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>,
                 UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>,
                 ConstructFakeOpPlaceHolder<tOut<FakeOpPlaceHolder>,
                                            tIn<List<Index>>>);

// Variable = Iterator | Index | FakeOpPlaceHolder
DEFINE_ADT_UNION(Variable,
                 Iterator,
                 Index,
                 FakeOpPlaceHolder);
// clang-format on

inline bool operator==(const Variable& lhs, const Variable& rhs) {
  return std::visit(
      [](auto&& lhs, auto&& rhs) {
        if constexpr (std::is_same<decltype(lhs), decltype(rhs)>::value) {
          return lhs == rhs;
        } else {
          return false;
        }
      },
      lhs,
      rhs);
}

inline bool operator==(const Iterator& lhs, const Iterator& rhs) {
  return lhs.value() == rhs.value();
}

inline bool operator==(const Index& lhs, const Index& rhs) {
  return lhs.value() == rhs.value();
}

inline bool operator==(const FakeOpPlaceHolder& lhs,
                       const FakeOpPlaceHolder& rhs) {
  return lhs.value() == rhs.value();
}

// Function = Equation
using Function = Equation;

}  // namespace cinn::adt::equation

namespace std {

template <>
struct hash<cinn::adt::equation::Iterator> final {
  std::size_t operator()(const cinn::adt::equation::Iterator& iterator) const {
    return iterator.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::equation::Index> final {
  std::size_t operator()(const cinn::adt::equation::Index& index) const {
    return index.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::equation::FakeOpPlaceHolder> final {
  std::size_t operator()(
      const cinn::adt::equation::FakeOpPlaceHolder& placeholder) const {
    return std::hash<std::string>()(placeholder.value());
  }
};

template <>
struct hash<cinn::adt::equation::Variable> final {
  std::size_t operator()(const cinn::adt::equation::Variable& variable) const {
    std::size_t hash_value =
        variable >>
        cinn::adt::match{
            [](const cinn::adt::equation::Iterator& iterator) {
              return std::hash<cinn::adt::equation::Iterator>()(iterator);
            },
            [](const cinn::adt::equation::Index& index) {
              return std::hash<cinn::adt::equation::Index>()(index);
            },
            [](const cinn::adt::equation::FakeOpPlaceHolder& placeholder) {
              return std::hash<cinn::adt::equation::FakeOpPlaceHolder>()(
                  placeholder);
            }};
    return cinn::adt::hash_combine(hash_value, variable.index());
  }
};

}  // namespace std
