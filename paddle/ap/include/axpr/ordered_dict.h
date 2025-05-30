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
#include <unordered_map>
#include <utility>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/hash.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename KeyT, typename ValueT, typename Hasher>
struct OrderedDictImpl {
 public:
  OrderedDictImpl() {}

  bool operator==(const OrderedDictImpl& other) const { return this == &other; }

  using ItemT = std::pair<KeyT, ValueT>;

  const std::list<ItemT>& items() const { return items_; }

  adt::Result<bool> Has(InterpreterBase<ValueT>* interpreter,
                        const KeyT& key) const {
    Hasher hasher{};
    ADT_LET_CONST_REF(hash_value, hasher(interpreter, key));
    const auto& iter_to_iters = this->hash_value2pair_iters_.find(hash_value);
    if (iter_to_iters == this->hash_value2pair_iters_.end()) {
      return false;
    }
    for (auto iter : iter_to_iters->second) {
      if (iter->first == key) {
        return true;
      }
    }
    return false;
  }

  adt::Result<ValueT> At(InterpreterBase<ValueT>* interpreter,
                         const KeyT& key) const {
    Hasher hasher{};
    ADT_LET_CONST_REF(hash_value, hasher(interpreter, key));
    const auto& iter_to_iters = this->hash_value2pair_iters_.find(hash_value);
    ADT_CHECK(iter_to_iters != this->hash_value2pair_iters_.end());
    for (auto iter : iter_to_iters->second) {
      if (iter->first == key) {
        return iter->second;
      }
    }
    return adt::errors::KeyError{"OrderedDictImpl::At() failed."};
  }

  adt::Result<adt::Ok> Insert(InterpreterBase<ValueT>* interpreter,
                              const ItemT& pair) {
    return Insert(interpreter, pair.first, pair.second);
  }

  adt::Result<adt::Ok> Insert(InterpreterBase<ValueT>* interpreter,
                              const ValueT& key,
                              const ValueT& val) {
    Hasher hasher{};
    ADT_LET_CONST_REF(hash_value, hasher(interpreter, key));
    auto* lst = &this->hash_value2pair_iters_[hash_value];
    for (auto iter : *lst) {
      if (iter->first == key) {
        iter->second = val;
        return adt::Ok{};
      }
    }
    lst->emplace_back(
        this->items_.insert(this->items_.end(), std::pair{key, val}));
    return adt::Ok{};
  }

 private:
  using ItemsT = std::list<ItemT>;
  ItemsT items_;
  std::unordered_map<int64_t, std::list<typename ItemsT::iterator>>
      hash_value2pair_iters_;
};

template <typename ValueT>
ADT_DEFINE_RC(OrderedDict, OrderedDictImpl<ValueT, ValueT, axpr::Hash<ValueT>>);

template <typename ValueT>
struct TypeImpl<OrderedDict<ValueT>> : public std::monostate {
  using std::monostate::monostate;

  const char* Name() const { return "OrderedDict"; }
};

}  // namespace ap::axpr
