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

#include <optional>
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/common/enforce.h"

namespace cinn::adt {

// Tree InnerT LeafT = LeafT | InnerT (Tree InnerT LeafT)
template <template <typename> class InnerT, typename LeafT>
DEFINE_ADT_UNION(Tree, LeafT, InnerT<Tree<InnerT, LeafT>>);

// TreeInnerNode T TreeT = (T, [TreeT])
template <typename T>
struct TreeInner {
  template <typename TreeT>
  struct Node final : public Tuple<T, List<TreeT>> {
    using value_type = T;
    using Tuple<T, List<TreeT>>::Tuple;
  };
};

template <typename T>
struct TreeTrait;

template <template <typename> class InnerT, typename LeafT>
struct TreeTrait<Tree<InnerT, LeafT>> {
  using inner_type = InnerT<Tree<InnerT, LeafT>>;
  using leaf_type = LeafT;
};

DEFINE_ADT_TAG(tCommon);
DEFINE_ADT_TAG(tLhsRemainder);
DEFINE_ADT_TAG(tRhsRemainder);

template <typename TreeT>
struct TreeMerger {
  using tree_type = TreeT;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;

  using inner_data_type = typename inner_type::value_type;
  inner_data_type GetInnerDataForLeaf(const leaf_type& leaf) const;

  inner_type MakeInnerNode(const inner_data_type& inner_data,
                           const List<TreeT>& children) const;

  using MergeResult = std::tuple<tCommon<inner_data_type>,
                                 tLhsRemainder<inner_data_type>,
                                 tRhsRemainder<inner_data_type>>;

  MergeResult MergeInnerValue(const inner_data_type& lhs,
                              const inner_data_type& rhs) const;
};

template <typename TreeMergerT>
List<typename TreeMergerT::tree_type> MergeTwoInnerTree(
    const TreeMergerT& tree_merger,
    const typename TreeMergerT::tree_type& lhs,
    const typename TreeMergerT::tree_type& rhs);

template <typename TreeMergerT>
List<typename TreeMergerT::tree_type> MergeTwoInnerTreeImpl(
    const TreeMergerT& tree_merger,
    const typename TreeTrait<typename TreeMergerT::tree_type>::inner_type& lhs,
    const typename TreeTrait<typename TreeMergerT::tree_type>::inner_type&
        rhs) {
  using TreeT = typename TreeMergerT::tree_type;
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using inner_data_type = typename inner_type::value_type;

  const auto& [lhs_inner_data, lhs_children] = lhs.tuple();
  const auto& [rhs_inner_data, rhs_children] = rhs.tuple();
  const auto& [common, lhs_remainder, rhs_remainder] =
      tree_merger.MergeInnerValue(lhs_inner_data, rhs_inner_data);

  bool is_common_empty = (lhs_remainder.value() == lhs_inner_data &&
                          rhs_remainder.value() == rhs_inner_data);
  if (is_common_empty) {
    return List<TreeT>{lhs, rhs};
  } else if (common.value() == lhs_inner_data &&
             common.value() == rhs_inner_data) {
    List<TreeT> merged_children{};
    merged_children->insert(
        merged_children->end(), lhs_children->begin(), lhs_children->end());
    merged_children->insert(
        merged_children->end(), rhs_children->begin(), rhs_children->end());
    const auto ret = tree_merger.MakeInnerNode(common.value(), merged_children);
    return List<TreeT>{ret};
  } else if (common.value() == lhs_inner_data &&
             common.value() != rhs_inner_data) {
    const auto new_rhs =
        tree_merger.MakeInnerNode(rhs_remainder.value(), rhs_children);
    const TreeT last_lhs_child = lhs_children->back();
    const auto merged_last_children =
        MergeTwoInnerTree(tree_merger, last_lhs_child, new_rhs);
    List<TreeT> new_lhs_children{};
    new_lhs_children->insert(new_lhs_children->end(),
                             lhs_children->begin(),
                             std::prev(lhs_children->end()));
    new_lhs_children->insert(new_lhs_children->end(),
                             merged_last_children->begin(),
                             merged_last_children->end());
    const auto ret =
        tree_merger.MakeInnerNode(common.value(), new_lhs_children);
    return List<TreeT>{ret};
  } else if (common.value() != lhs_inner_data &&
             common.value() == rhs_inner_data) {
    const auto new_lhs =
        tree_merger.MakeInnerNode(lhs_remainder.value(), lhs_children);
    const TreeT first_rhs_child = *rhs_children->begin();
    const auto merged_first_children =
        MergeTwoInnerTree(tree_merger, new_lhs, first_rhs_child);
    List<TreeT> new_rhs_children = merged_first_children;
    new_rhs_children->insert(new_rhs_children->end(),
                             std::next(rhs_children->begin()),
                             rhs_children->end());
    const auto ret =
        tree_merger.MakeInnerNode(common.value(), new_rhs_children);
    return List<TreeT>{ret};
  } else if (common.value() != lhs_inner_data &&
             common.value() != rhs_inner_data) {
    const auto new_lhs =
        tree_merger.MakeInnerNode(lhs_remainder.value(), lhs_children);
    const auto new_rhs =
        tree_merger.MakeInnerNode(rhs_remainder.value(), rhs_children);
    const auto ret = tree_merger.MakeInnerNode(common.value(),
                                               List<TreeT>{new_lhs, new_rhs});
    return List<TreeT>{ret};
  } else {
    PADDLE_THROW(::common::errors::Fatal("Dead code"));
  }
}

template <typename TreeMergerT>
List<typename TreeMergerT::tree_type> MergeTwoInnerTree(
    const TreeMergerT& tree_merger,
    const typename TreeMergerT::tree_type& lhs,
    const typename TreeMergerT::tree_type& rhs) {
  using TreeT = typename TreeMergerT::tree_type;
  using inner_type = typename TreeTrait<TreeT>::inner_type;

  return std::visit(
      [&](const auto& lhs, const auto& rhs) -> List<TreeT> {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, inner_type> &&
                      std::is_same_v<std::decay_t<decltype(rhs)>, inner_type>) {
          return MergeTwoInnerTreeImpl(tree_merger, lhs, rhs);
        } else {
          return List<TreeT>{lhs, rhs};
        }
      },
      lhs.variant(),
      rhs.variant());
}

template <typename TreeMergerT>
void MergeTrees(
    const TreeMergerT& tree_merger,
    List<typename TreeMergerT::tree_type>* acc,
    const List<typename TreeTrait<typename TreeMergerT::tree_type>::leaf_type>&
        leaves) {
  using TreeT = typename TreeMergerT::tree_type;
  if (leaves->empty()) {
    return;
  }
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using inner_data_type = typename inner_type::value_type;

  const auto& MakeTreeFromLeaf = [&](const leaf_type& leaf) -> TreeT {
    const inner_data_type inner_data = tree_merger.GetInnerDataForLeaf(leaf);
    const auto ret =
        tree_merger.MakeInnerNode(inner_data, List<TreeT>{TreeT{leaf}});
    return ret;
  };

  // Handle init
  std::size_t leaf_start = 0;
  if ((*acc)->empty()) {
    (*acc)->emplace_back(MakeTreeFromLeaf(leaves->at(0)));
    leaf_start = 1;
  }

  for (std::size_t i = leaf_start; i < leaves->size(); ++i) {
    const auto merged = MergeTwoInnerTree(
        tree_merger, (*acc)->back(), MakeTreeFromLeaf(leaves->at(i)));
    (*acc)->erase(std::prev((*acc)->end()));
    (*acc)->insert((*acc)->end(), merged->begin(), merged->end());
  }
}

template <typename TreeMergerT>
List<typename TreeMergerT::tree_type> MakeMergedTrees(
    const TreeMergerT& tree_merger,
    const List<typename TreeTrait<typename TreeMergerT::tree_type>::leaf_type>&
        leaves) {
  using TreeT = typename TreeMergerT::tree_type;

  List<TreeT> acc{};
  MergeTrees(tree_merger, &acc, leaves);
  return acc;
}

}  // namespace cinn::adt
