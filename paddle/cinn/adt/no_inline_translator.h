#pragma once

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/cinn/adt/inline_translator_trait.h"

namespace cinn::adt {

template <template <typename> class MapT,
          template <typename>
          class OpCallT,
          typename TensorT>
struct NoInlineTranslator final {
  using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  using OpExpr = Tree<OpCallT, Load<TensorT>>;
  using DstLeaf = Store<TensorT, OpExpr>;
  using SrcTree = Tree<MapT, SrcLeaf>;
  using DstTree = Tree<MapT, DstLeaf>;

  static DstTree Call(const SrcTree& src_tree) {
    return Translate(src_tree);
  }

 private:
  static DstTree Translate(const SrcTree& src_tree) {
    return std::visit([&](const auto& impl){
      return TranslateImpl(impl);
    }, src_tree.variant());
  }

  static DstTree TranslateImpl(const MapT<SrcTree>& src_map) {
    return DstTree{TranslateMap(src_map)};
  }

  static MapT<DstTree> TranslateMap(const MapT<SrcTree>& src_map) {
    const List<SrcTree> src_children =
        InlineTranslatorTrait<MapT>::GetTreeInnerNodeChildren(src_map);
    const List<DstTree> dst_children = TranslateList(src_children);
    return InlineTranslatorTrait<MapT>::ConvertMap(src_map, dst_children);
  }

  static List<DstTree> TranslateList(const List<SrcTree>& src_children) {
    List<DstTree> ret{};
    for (const auto& src_child : *src_children) {
      ret->emplace_back(Translate(src_child));
    }
    return ret;
  }

  static DstTree TranslateImpl(const SrcLeaf& src_leaf) {
    return DstTree{TranslateLeaf(src_leaf)};
  }

  // using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  // using DstLeaf = Store<TensorT, OpExpr>;
  static DstLeaf TranslateLeaf(const SrcLeaf& src_leaf) {
    const auto& [tensor, op_call] = src_leaf.tuple();
    const List<Load<TensorT>>& src_loads =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    List<OpExpr> dst_loads{};
    for (const auto& src_load : *src_loads) {
      dst_loads->emplace_back(src_load);
    }
    OpCallT<OpExpr> dst_op_call =
        InlineTranslatorTrait<OpCallT>::ConvertMap(op_call, dst_loads);
    OpExpr dst_op_call_tree = dst_op_call;
    return DstLeaf{tensor, dst_op_call_tree};
  }

};

}
