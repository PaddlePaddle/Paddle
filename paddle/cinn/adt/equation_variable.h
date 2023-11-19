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

#include <functional>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/adt/unique_id.h"

namespace cinn::adt {

// Iterator = tIterator UniqueId
using Iterator = tIterator<UniqueId>;
// IteratorTuple = [Iterator]
using IteratorTuple = List<Iterator>;
// Index = tIndex UniqueId
using Index = tIndex<UniqueId>;
// FakeOpPlaceHolder = tOpPlaceHolder UniqueId
using FakeOpPlaceHolder = tOpPlaceHolder<UniqueId>;

// Variable = Iterator | Index | FakeOpPlaceHolder
DEFINE_ADT_UNION(Variable, Iterator, Index, FakeOpPlaceHolder);

OVERLOAD_OPERATOR_EQ_NE(Variable, UnionEqual);

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
