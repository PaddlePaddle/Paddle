#pragma once

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/m_expr.h"

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


}
