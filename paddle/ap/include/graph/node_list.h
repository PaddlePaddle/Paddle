// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <variant>
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::graph {

template <typename T>
struct Node;

template <typename T>
struct UndefinedTag : public std::monostate {
  using std::monostate::monostate;
};

template <typename T>
struct IndexedTag {
  T data;
};

template <typename T>
struct UnindexedTag {
  T data;
};

template <typename T>
using ValidListTagImpl = std::variant<IndexedTag<T>, UnindexedTag<T>>;

template <typename T>
struct ValidListTag : public ValidListTagImpl<T> {
  using ValidListTagImpl<T>::ValidListTagImpl;
  ADT_DEFINE_VARIANT_METHODS(ValidListTagImpl<T>);
};

template <typename T>
using ListTagImpl =
    std::variant<UndefinedTag<T>, IndexedTag<T>, UnindexedTag<T>>;

template <typename T>
struct ListTag : public ListTagImpl<T> {
  using ListTagImpl<T>::ListTagImpl;
  ADT_DEFINE_VARIANT_METHODS(ListTagImpl<T>);
};

template <typename T>
struct NodeList : public ListTag<adt::List<Node<T>>> {
  using list_type = adt::List<Node<T>>;

  using ListTag<list_type>::ListTag;

  ListTag<std::monostate> type() const {
    return this->Match(
        [](const UndefinedTag<list_type>&) -> ListTag<std::monostate> {
          return UndefinedTag<std::monostate>{};
        },
        [](const IndexedTag<list_type>&) -> ListTag<std::monostate> {
          return IndexedTag<std::monostate>{};
        },
        [](const UnindexedTag<list_type>&) -> ListTag<std::monostate> {
          return UnindexedTag<std::monostate>{};
        });
  }

  adt::Result<Node<T>> Sole() const {
    return this->Match(
        [](const UndefinedTag<list_type>&) -> adt::Result<Node<T>> {
          return adt::errors::TypeError{"UndefinedList has no sole data"};
        },
        [](const auto& l) -> adt::Result<Node<T>> {
          ADT_CHECK(l.data->size() == 1);
          return l.data->at(0);
        });
  }

  std::size_t size() const {
    return this->Match(
        [](const UndefinedTag<list_type>&) -> std::size_t { return 0; },
        [](const auto& l) -> std::size_t { return l.data->size(); });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitNodes(const DoEachT& DoEach) const {
    return this->Match(
        [](const UndefinedTag<list_type>&) -> adt::Result<adt::Ok> {
          return adt::Ok{};
        },
        [&](const auto& l) -> adt::Result<adt::Ok> {
          for (const auto& data : *l.data) {
            ADT_RETURN_IF_ERR(DoEach(data));
          }
          return adt::Ok{};
        });
  }
};

}  // namespace ap::graph
