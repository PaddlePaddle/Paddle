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

#include <functional>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/drr/native_ir_value.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/packed_ir_value.h"

namespace ap::drr {

using IrValueImpl =
    std::variant<NativeIrValue<drr::Node>, PackedIrValue<drr::Node>>;

struct IrValue : public IrValueImpl {
  using IrValueImpl::IrValueImpl;
  ADT_DEFINE_VARIANT_METHODS(IrValueImpl);

  const graph::Node<drr::Node>& node() const {
    return Match([](const auto& impl) -> const graph::Node<drr::Node>& {
      return impl->node;
    });
  }

  static std::optional<IrValue> OptCastFrom(const drr::Node& drr_node) {
    using RetT = std::optional<IrValue>;
    return drr_node.Match(
        [](const NativeIrValue<drr::Node>& ir_value) -> RetT {
          return IrValue{ir_value};
        },
        [](const PackedIrValue<drr::Node>& ir_value) -> RetT {
          return IrValue{ir_value};
        },
        [](const auto&) -> RetT { return std::nullopt; });
  }

  const std::string& name() const {
    return Match(
        [](const auto& impl) -> const std::string& { return impl->name; });
  }
};

}  // namespace ap::drr

namespace std {

template <>
struct hash<ap::drr::IrValue> {
  std::size_t operator()(const ap::drr::IrValue& ir_value) const {
    return ir_value.Match([&](const auto& impl) -> std::size_t {
      return reinterpret_cast<std::size_t>(impl.__adt_rc_shared_ptr_raw_ptr());
    });
  }
};

}  // namespace std
