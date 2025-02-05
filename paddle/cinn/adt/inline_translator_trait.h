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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/map_expr.h"

namespace cinn::adt {

template <template <typename> class MapT>
struct InlineTranslatorTrait;

template <>
struct InlineTranslatorTrait<MapStmt> final {
  template <typename T>
  static List<T> GetTreeInnerNodeChildren(const MapStmt<T>& map_stmt) {
    const auto& [iterators, stmts] = map_stmt.tuple();
    return stmts;
  }

  template <typename SrcTreeT, typename DstTreeT>
  static MapStmt<DstTreeT> ConvertMap(const MapStmt<SrcTreeT>& src_map,
                                      const List<DstTreeT>& dst_children) {
    const auto& [iterators, src_children] = src_map.tuple();
    return MapStmt<DstTreeT>{iterators, dst_children};
  }
};

// OpCall T = (Op, [T])
template <>
struct InlineTranslatorTrait<OpCall> final {
  template <typename T>
  static List<T> GetTreeInnerNodeChildren(const OpCall<T>& op_call) {
    const auto& [op, tensors] = op_call.tuple();
    return tensors;
  }

  template <typename SrcTreeT, typename DstTreeT>
  static OpCall<DstTreeT> ConvertMap(const OpCall<SrcTreeT>& src_map,
                                     const List<DstTreeT>& dst_children) {
    const auto& [op, _] = src_map.tuple();
    return OpCall<DstTreeT>{op, dst_children};
  }
};

}  // namespace cinn::adt
