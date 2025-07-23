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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/list.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/mutable_list.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

template <typename ValueT>
using AbstractListImpl = std::variant<adt::List<ValueT>,
                                      adt::List<SerializableValue>,
                                      axpr::MutableList<ValueT>>;

template <typename ValueT>
struct AbstractList : public AbstractListImpl<ValueT> {
  using AbstractListImpl<ValueT>::AbstractListImpl;

  ADT_DEFINE_VARIANT_METHODS(AbstractListImpl<ValueT>);

  static adt::Result<AbstractList<ValueT>> CastFrom(const ValueT& value) {
    using RetT = adt::Result<AbstractList<ValueT>>;
    return value.Match(
        [&](const adt::List<ValueT>& impl) -> RetT { return impl; },
        [&](const adt::List<SerializableValue>& impl) -> RetT { return impl; },
        [&](const axpr::MutableList<ValueT>& impl) -> RetT { return impl; },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "only list, SerializableList, MutableList are convertible to "
              "AbstractList. (" +
              GetTypeName(value) + " given)"};
        });
  }

  static bool CastableFrom(const ValueT& value) {
    using RetT = bool;
    return value.Match(
        [&](const adt::List<ValueT>& impl) -> RetT { return true; },
        [&](const adt::List<SerializableValue>& impl) -> RetT { return true; },
        [&](const axpr::MutableList<ValueT>& impl) -> RetT { return true; },
        [&](const auto&) -> RetT { return false; });
  }

  adt::Result<std::size_t> size() const {
    using RetT = adt::Result<std::size_t>;
    return Match(
        [](const axpr::MutableList<ValueT>& impl) -> RetT {
          ADT_LET_CONST_REF(data_vec, impl.Get());
          return data_vec->size();
        },
        [](const auto& impl) -> RetT { return impl->size(); });
  }

  adt::Result<ValueT> at(std::size_t i) const {
    using RetT = adt::Result<ValueT>;
    return Match(
        [&](const adt::List<ValueT>& impl) -> RetT { return impl->at(i); },
        [&](const adt::List<SerializableValue>& impl) -> RetT {
          return impl->at(i).template CastTo<ValueT>();
        },
        [&](const axpr::MutableList<ValueT>& impl) -> RetT {
          ADT_LET_CONST_REF(data_vec, impl.Get());
          return data_vec->at(i);
        });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> Visit(const DoEachT& DoEach) const {
    using Ok = adt::Result<adt::Ok>;
    return Match(
        [&](const adt::List<ValueT>& impl) -> Ok {
          for (const auto& elt : *impl) {
            ADT_LET_CONST_REF(loop_ctrl, DoEach(elt));
            if (loop_ctrl.template Has<adt::Break>()) {
              break;
            }
          }
          return adt::Ok{};
        },
        [&](const adt::List<SerializableValue>& impl) -> Ok {
          for (const auto& serializable_elt : *impl) {
            const auto& elt = serializable_elt.template CastTo<ValueT>();
            ADT_LET_CONST_REF(loop_ctrl, DoEach(elt));
            if (loop_ctrl.template Has<adt::Break>()) {
              break;
            }
          }
          return adt::Ok{};
        },
        [&](const axpr::MutableList<ValueT>& impl) -> Ok {
          ADT_LET_CONST_REF(vec, impl.Get());
          for (const auto& elt : *vec) {
            ADT_LET_CONST_REF(loop_ctrl, DoEach(elt));
            if (loop_ctrl.template Has<adt::Break>()) {
              break;
            }
          }
          return adt::Ok{};
        });
  }
};

}  // namespace ap::axpr
