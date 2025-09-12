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
#include <unordered_map>
#include <vector>
#include "paddle/ap/include/adt/adt.h"

namespace ap::ir_match {

template <typename IrValueT, typename IrOperandT>
struct RefNodeInfoImpl {
  IrValueT ir_value;
  adt::List<IrOperandT> op_operands_subset;

  bool operator==(const RefNodeInfoImpl& other) const { return this == &other; }
};

template <typename IrValueT, typename IrOperandT>
ADT_DEFINE_RC(RefNodeInfo, RefNodeInfoImpl<IrValueT, IrOperandT>);

}  // namespace ap::ir_match

namespace std {

template <typename IrValueT, typename IrOperandT>
struct hash<ap::ir_match::RefNodeInfo<IrValueT, IrOperandT>> {
  std::size_t operator()(
      const ap::ir_match::RefNodeInfo<IrValueT, IrOperandT>& node) const {
    return reinterpret_cast<std::size_t>(node.shared_ptr().get());
  }
};

}  // namespace std
