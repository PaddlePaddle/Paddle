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
#include "paddle/ap/include/axpr/class_instance.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

template <typename ValueT, typename ItemValueT>
struct ClassAttrsHelper {
  std::optional<ValueT> OptGet(const ClassAttrs<ItemValueT>& class_attrs,
                               const std::string& attr_name) {
    return ImplOptGet(class_attrs.shared_ptr().get(), attr_name);
  }

  std::optional<ValueT> OptGet(const ClassAttrsImpl<ItemValueT>* class_attrs,
                               const std::string& attr_name) {
    return ImplOptGet(class_attrs, attr_name);
  }

 private:
  std::optional<ValueT> ImplOptGet(
      const ClassAttrsImpl<ItemValueT>* class_attrs_impl,
      const std::string& attr_name) {
    const auto& opt_val = class_attrs_impl->attrs->OptGet(attr_name);
    if (opt_val.has_value()) {
      if constexpr (std::is_same_v<ValueT, ItemValueT>) {
        return opt_val.value();
      } else {
        return opt_val.value().template CastTo<ValueT>();
      }
    }
    for (const auto& base : *class_attrs_impl->superclasses) {
      if (const auto val_in_base = ImplOptGet(base.get(), attr_name)) {
        return val_in_base.value();
      }
    }
    return std::nullopt;
  }
};

}  // namespace ap::axpr
