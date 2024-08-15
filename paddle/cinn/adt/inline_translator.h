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
struct InlineTranslator final {
  using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  using OpExpr = Tree<OpCallT, Load<TensorT>>;
  using DstLeaf = Store<TensorT, OpExpr>;
  using SrcTree = Tree<MapT, SrcLeaf>;
  using DstTree = Tree<MapT, DstLeaf>;

  static DstTree Call(const SrcTree& src_tree) {
    PADDLE_ENFORCE_EQ((src_tree.template Has<MapT<SrcTree>>()),
                      true,
                      ::common::errors::InvalidArgument(
                          "src_tree.template should have <MapT<SrcTree>>()"));
    const MapT<DstTree> dst_tree =
        CallMap(src_tree.template Get<MapT<SrcTree>>());

    return DstTree{dst_tree};
  }

 private:
  static MapT<DstTree> CallMap(const MapT<SrcTree>& src_map) {
    const List<SrcTree> src_children =
        InlineTranslatorTrait<MapT>::GetTreeInnerNodeChildren(src_map);
    const List<DstTree> dst_children = CallList(src_children);
    return InlineTranslatorTrait<MapT>::ConvertMap(src_map, dst_children);
  }

  static List<DstTree> CallList(const List<SrcTree>& src_children) {
    List<DstTree> ret{};

    VisitEachContiguousSegment(
        src_children, [&](int start, int end, bool is_leaf) {
          if (!is_leaf) {
            for (int i = start; i < end; ++i) {
              ret->emplace_back(Call(src_children->at(i)));
            }
          } else {
            const auto& converted = TranslateContiguousLeaves(
                std::next(src_children->begin(), start),
                std::next(src_children->begin(), end));
            ret->insert(ret->end(), converted->begin(), converted->end());
          }
        });

    return ret;
  }

  struct ConsumerPos {
    int leaf_index;
    int arg_index;
  };

  // using DstLeaf = Store<TensorT, OpExpr>;
  static DstLeaf UpdateConsumerArg(const DstLeaf& consumer,
                                   int arg_index,
                                   const DstLeaf& producer) {
    const auto& [consumer_tensor, consumer_tree] = consumer.tuple();
    CheckConsumerPosIsLoadTensor(consumer, arg_index);
    const auto& op_call = consumer_tree.template Get<OpCallT<OpExpr>>();
    const auto& op_call_children =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    const auto& ret_op_call_children =
        UpdateConsumerArg(op_call_children, arg_index, producer);
    const auto& ret_op_call = InlineTranslatorTrait<OpCallT>::ConvertMap(
        op_call, ret_op_call_children);
    OpExpr ret_op_call_tree = ret_op_call;
    return DstLeaf{consumer_tensor, ret_op_call_tree};
  }

  static List<OpExpr> UpdateConsumerArg(const List<OpExpr>& op_call_children,
                                        int arg_index,
                                        const DstLeaf& producer) {
    const auto& [producer_tensor, producer_tree] = producer.tuple();
    const auto& arg = op_call_children->at(arg_index);
    const auto& arg_leaf = arg.template Get<Load<TensorT>>();
    const auto& [arg_tensor] = arg_leaf.tuple();
    PADDLE_ENFORCE_EQ(producer_tensor == arg_tensor,
                      true,
                      ::common::errors::InvalidArgument(
                          "producer_tensor should be equal to arg_tensor"));
    List<OpExpr> ret{};
    ret->assign(op_call_children->begin(), op_call_children->end());
    ret->at(arg_index) = producer_tree;
    return ret;
  }

  // using DstLeaf = Store<TensorT, OpExpr>;
  static void CheckConsumerPosIsLoadTensor(const DstLeaf& consumer,
                                           int arg_index) {
    const auto& [tensor, consumer_tree] = consumer.tuple();
    PADDLE_ENFORCE_EQ(
        (consumer_tree.template Has<OpCallT<OpExpr>>()),
        true,
        ::common::errors::InvalidArgument(
            "consumer_tree.template should have <OpCallT<OpExpr>>()"));
    const auto& op_call = consumer_tree.template Get<OpCallT<OpExpr>>();
    const auto& op_call_children =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    const auto& op_call_child = op_call_children->at(arg_index);
    PADDLE_ENFORCE_EQ(
        (op_call_child.template Has<Load<TensorT>>()),
        true,
        ::common::errors::InvalidArgument(
            "op_call_child.template should have <Load<TensorT>>()"));
  }

  template <typename DoEachT>
  static void VisitEachArg(const SrcTree& tree, const DoEachT& DoEach) {
    const auto& [_, op_call] = tree.template Get<SrcLeaf>().tuple();
    const auto& args =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    for (int i = 0; i < args->size(); ++i) {
      const auto& [tensor] = args->at(i).tuple();
      DoEach(tensor, i);
    }
  }

  // using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  template <typename SrcTreeIterT>
  static std::vector<std::vector<ConsumerPos>> MakeProducerIndex2ConsumerPos(
      SrcTreeIterT begin, SrcTreeIterT end) {
    std::vector<std::vector<ConsumerPos>> producer_index2consumer_positions(
        end - begin);
    for (SrcTreeIterT producer = begin; producer != end; ++producer) {
      const auto& [producer_tensor, _] =
          (*producer).template Get<SrcLeaf>().tuple();
      for (SrcTreeIterT consumer = std::next(producer); consumer != end;
           ++consumer) {
        VisitEachArg(*consumer, [&](const TensorT arg_tensor, int arg_idx) {
          if (arg_tensor == producer_tensor) {
            auto* vec = &producer_index2consumer_positions.at(producer - begin);
            vec->push_back(ConsumerPos{.leaf_index = consumer - begin,
                                       .arg_index = arg_idx});
          }
        });
      }
    }
    return producer_index2consumer_positions;
  }

  template <typename SrcTreeIterT>
  static List<DstTree> TranslateContiguousLeaves(SrcTreeIterT begin,
                                                 SrcTreeIterT end) {
    int size = end - begin;
    const auto producer_idx2consumer_pos =
        MakeProducerIndex2ConsumerPos(begin, end);
    const auto& GetConsumerPos4ProducerIndex =
        [&](int index) -> std::vector<ConsumerPos> {
      return producer_idx2consumer_pos.at(index);
    };
    std::unordered_map<int, DstLeaf> index2dst_leaf{};
    // Init dst leaves
    for (int i = 0; i < size; ++i) {
      PADDLE_ENFORCE_EQ(
          index2dst_leaf.emplace(i, NaiveTranslateLeaf(*std::next(begin, i)))
              .second,
          true,
          ::common::errors::InvalidArgument(
              "index2dst_leaf.emplace should return true"));
    }
    // Inline dst leaves
    for (int producer_i = 0; producer_i < size; ++producer_i) {
      const auto& consumer_positions = GetConsumerPos4ProducerIndex(producer_i);
      if (consumer_positions.empty()) {
        // Do nothing
      } else {
        DstLeaf producer = index2dst_leaf.at(producer_i);
        for (const auto& consumer_pos : consumer_positions) {
          DstLeaf consumer = index2dst_leaf.at(consumer_pos.leaf_index);
          index2dst_leaf.at(consumer_pos.leaf_index) =
              UpdateConsumerArg(consumer, consumer_pos.arg_index, producer);
        }
        index2dst_leaf.erase(producer_i);
      }
    }
    // Collect inlined leaves
    List<DstTree> ret{};
    for (int i = 0; i < size; ++i) {
      const auto& iter = index2dst_leaf.find(i);
      if (iter != index2dst_leaf.end()) {
        ret->emplace_back(iter->second);
      }
    }
    return ret;
  }

  // using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  // using DstLeaf = Store<TensorT, OpExpr>;
  static DstLeaf NaiveTranslateLeaf(const SrcTree& src_tree) {
    PADDLE_ENFORCE_EQ(src_tree.template Has<SrcLeaf>(),
                      true,
                      ::common::errors::InvalidArgument(
                          "src_tree.template should have <SrcLeaf>()"));
    const auto& [tensor, op_call] = src_tree.template Get<SrcLeaf>().tuple();
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

  template <typename DoEachT /*void(&)(int start, int end, bool is_leaf)*/>
  static void VisitEachContiguousSegment(const List<SrcTree>& src_children,
                                         const DoEachT& DoEach) {
    std::vector<int> child_index2is_leaf(src_children->size(), 0);
    for (int i = 0; i < src_children->size(); ++i) {
      child_index2is_leaf.at(i) = src_children->at(i).template Has<SrcLeaf>();
    }
    int start = 0;
    for (int i = 1; i < child_index2is_leaf.size(); ++i) {
      if (child_index2is_leaf.at(i - 1) != child_index2is_leaf.at(i)) {
        DoEach(start, i, child_index2is_leaf.at(i - 1));
        start = i;
      } else {
        // Do nothing
      }
    }
    if (start != child_index2is_leaf.size()) {
      DoEach(start, child_index2is_leaf.size(), child_index2is_leaf.back());
    }
  }
};

}  // namespace cinn::adt
