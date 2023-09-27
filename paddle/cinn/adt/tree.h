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
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;

  using inner_data_type = typename inner_type::value_type;
  static inner_data_type GetInnerDataForLeaf(const leaf_type& leaf);

  static inner_type MakeInnerNode(const inner_data_type& inner_data,
                                  const List<TreeT>& children);

  using MergeResult = std::tuple<tCommon<inner_data_type>,
                                 tLhsRemainder<inner_data_type>,
                                 tRhsRemainder<inner_data_type>>;

  static MergeResult MergeInnerValue(const inner_data_type& lhs,
                                     const inner_data_type& rhs);
};

template <typename TreeT>
List<TreeT> MergeTwoInnerTree(const TreeT& lhs, const TreeT& rhs);

template <typename TreeT>
List<TreeT> MergeTwoInnerTreeImpl(
    const typename TreeTrait<TreeT>::inner_type& lhs,
    const typename TreeTrait<TreeT>::inner_type& rhs) {
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using inner_data_type = typename inner_type::value_type;

  const auto& [lhs_inner_data, lhs_children] = lhs.tuple();
  const auto& [rhs_inner_data, rhs_children] = rhs.tuple();
  const auto& [common, lhs_remainder, rhs_remainder] =
      TreeMerger<TreeT>::MergeInnerValue(lhs_inner_data, rhs_inner_data);

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
    const auto ret =
        TreeMerger<TreeT>::MakeInnerNode(common.value(), merged_children);
    return List<TreeT>{ret};
  } else if (common.value() == lhs_inner_data &&
             common.value() != rhs_inner_data) {
    const auto new_rhs =
        TreeMerger<TreeT>::MakeInnerNode(rhs_remainder.value(), rhs_children);
    const TreeT last_lhs_child = lhs_children->back();
    const auto merged_last_children =
        MergeTwoInnerTree<TreeT>(last_lhs_child, new_rhs);
    List<TreeT> new_lhs_children{};
    new_lhs_children->insert(new_lhs_children->end(),
                             lhs_children->begin(),
                             std::prev(lhs_children->end()));
    new_lhs_children->insert(new_lhs_children->end(),
                             merged_last_children->begin(),
                             merged_last_children->end());
    const auto ret =
        TreeMerger<TreeT>::MakeInnerNode(common.value(), new_lhs_children);
    return List<TreeT>{ret};
  } else if (common.value() != lhs_inner_data &&
             common.value() == rhs_inner_data) {
    const auto new_lhs =
        TreeMerger<TreeT>::MakeInnerNode(lhs_remainder.value(), lhs_children);
    const TreeT first_rhs_child = *rhs_children->begin();
    const auto merged_first_children =
        MergeTwoInnerTree<TreeT>(new_lhs, first_rhs_child);
    List<TreeT> new_rhs_children = merged_first_children;
    new_rhs_children->insert(new_rhs_children->end(),
                             std::next(rhs_children->begin()),
                             rhs_children->end());
    const auto ret =
        TreeMerger<TreeT>::MakeInnerNode(common.value(), new_rhs_children);
    return List<TreeT>{ret};
  } else if (common.value() != lhs_inner_data &&
             common.value() != rhs_inner_data) {
    const auto new_lhs =
        TreeMerger<TreeT>::MakeInnerNode(lhs_remainder.value(), lhs_children);
    const auto new_rhs =
        TreeMerger<TreeT>::MakeInnerNode(rhs_remainder.value(), rhs_children);
    const auto ret = TreeMerger<TreeT>::MakeInnerNode(
        common.value(), List<TreeT>{new_lhs, new_rhs});
    return List<TreeT>{ret};
  } else {
    LOG(FATAL) << "Dead code";
  }
}

template <typename TreeT>
List<TreeT> MergeTwoInnerTree(const TreeT& lhs, const TreeT& rhs) {
  using inner_type = typename TreeTrait<TreeT>::inner_type;

  return std::visit(
      [&](const auto& lhs, const auto& rhs) -> List<TreeT> {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, inner_type> &&
                      std::is_same_v<std::decay_t<decltype(rhs)>, inner_type>) {
          return MergeTwoInnerTreeImpl<TreeT>(lhs, rhs);
        } else {
          return List<TreeT>{lhs, rhs};
        }
      },
      lhs.variant(),
      rhs.variant());
}

template <typename TreeT>
List<TreeT> MakeTreeByMerger(
    const List<typename TreeTrait<TreeT>::leaf_type>& leaves) {
  if (leaves->empty()) {
    return List<TreeT>{};
  }
  using leaf_type = typename TreeTrait<TreeT>::leaf_type;
  using inner_type = typename TreeTrait<TreeT>::inner_type;
  using inner_data_type = typename inner_type::value_type;

  const auto& MakeTreeFromLeaf = [&](const leaf_type& leaf) -> TreeT {
    const inner_data_type inner_data =
        TreeMerger<TreeT>::GetInnerDataForLeaf(leaf);
    const auto ret =
        TreeMerger<TreeT>::MakeInnerNode(inner_data, List<TreeT>{TreeT{leaf}});
    return ret;
  };

  const auto& Aggregate = [&](const TreeT& init,
                              const auto& Merge) -> List<TreeT> {
    List<TreeT> acc{init};
    for (std::size_t i = 1; i < leaves->size(); ++i) {
      const auto merged = Merge(acc->back(), MakeTreeFromLeaf(leaves->at(i)));
      acc->erase(std::prev(acc->end()));
      acc->insert(acc->end(), merged->begin(), merged->end());
    }
    return acc;
  };

  return Aggregate(MakeTreeFromLeaf(leaves->at(0)), MergeTwoInnerTree<TreeT>);
}

}  // namespace cinn::adt
