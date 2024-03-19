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
#include "paddle/cinn/adt/inline_translator_trait.h"
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/adt/tree.h"

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

  static DstTree Call(const SrcTree& src_tree) { return Translate(src_tree); }

 private:
  static DstTree Translate(const SrcTree& src_tree) {
    return std::visit([&](const auto& impl) { return TranslateImpl(impl); },
                      src_tree.variant());
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

}  // namespace cinn::adt
