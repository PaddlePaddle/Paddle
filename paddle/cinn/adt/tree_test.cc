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

#include "paddle/cinn/adt/tree.h"

#include "gtest/gtest.h"

namespace cinn::adt {

namespace test {

using IntTreeLeafT = std::vector<int>;
using IntTreeInnerDataT = std::vector<int>;
using IntVecTree = Tree<TreeInner<IntTreeInnerDataT>::Node, IntTreeLeafT>;
using IntTreeInnerT = TreeInner<IntTreeInnerDataT>::template Node<IntVecTree>;

}  // namespace test

template <>
struct TreeMerger<test::IntVecTree> {
  using tree_type = test::IntVecTree;
  using inner_type = typename TreeTrait<test::IntVecTree>::inner_type;
  using leaf_type = typename TreeTrait<test::IntVecTree>::leaf_type;
  using inner_data_type = typename inner_type::value_type;

  inner_data_type GetInnerDataForLeaf(const leaf_type& leaf) const {
    return leaf;
  }

  inner_type MakeInnerNode(const inner_data_type& inner_data,
                           const List<test::IntVecTree>& children) const {
    return inner_type{inner_data, children};
  }

  using MergeResult = std::tuple<tCommon<inner_data_type>,
                                 tLhsRemainder<inner_data_type>,
                                 tRhsRemainder<inner_data_type>>;

  MergeResult MergeInnerValue(const inner_data_type& lhs,
                              const inner_data_type& rhs) const {
    inner_data_type common{};
    inner_data_type lhs_remainder{};
    inner_data_type rhs_remainder{};
    int min_size = std::min(lhs.size(), rhs.size());
    int idx = 0;
    for (; idx < min_size; ++idx) {
      if (lhs.at(idx) == rhs.at(idx)) {
        common.emplace_back(lhs.at(idx));
      } else {
        break;
      }
    }
    for (int lhs_idx = idx; lhs_idx < lhs.size(); ++lhs_idx) {
      lhs_remainder.emplace_back(lhs.at(lhs_idx));
    }
    for (int rhs_idx = idx; rhs_idx < rhs.size(); ++rhs_idx) {
      rhs_remainder.emplace_back(rhs.at(rhs_idx));
    }
    return MergeResult{common, lhs_remainder, rhs_remainder};
  }
};

namespace test {

TEST(IntVecTree, naive) {
  List<IntTreeLeafT> leaves{IntTreeLeafT{1, 2, 3}, IntTreeLeafT{4, 5, 6}};
  TreeMerger<test::IntVecTree> tree_merger{};
  List<IntVecTree> ret = MakeMergedTrees(tree_merger, leaves);
  ASSERT_EQ(ret->size(), 2);

  ASSERT_TRUE(ret->at(0).Has<IntTreeInnerT>());
  const auto& [inner_data0, children0] =
      ret->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data0 == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->size() == 1));
  ASSERT_TRUE((children0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children0->at(0).Get<IntTreeLeafT>() == IntTreeLeafT{1, 2, 3}));

  ASSERT_TRUE(ret->at(1).Has<IntTreeInnerT>());
  const auto& [inner_data1, children1] =
      ret->at(1).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data1 == IntTreeLeafT{4, 5, 6}));
  ASSERT_TRUE((children1->size() == 1));
  ASSERT_TRUE((children1->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children1->at(0).Get<IntTreeLeafT>() == IntTreeLeafT{4, 5, 6}));
}

TEST(IntVecTree, left_equal_right) {
  List<IntTreeLeafT> leaves{IntTreeLeafT{1, 2, 3}, IntTreeLeafT{1, 2, 3}};
  List<IntVecTree> ret =
      MakeMergedTrees(TreeMerger<test::IntVecTree>{}, leaves);
  ASSERT_EQ(ret->size(), 1);

  ASSERT_TRUE(ret->at(0).Has<IntTreeInnerT>());
  const auto& [inner_data0, children0] =
      ret->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data0 == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->size() == 2));
  ASSERT_TRUE((children0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children0->at(0).Get<IntTreeLeafT>() == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->at(1).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children0->at(1).Get<IntTreeLeafT>() == IntTreeLeafT{1, 2, 3}));
}

TEST(IntVecTree, left_gt_right) {
  List<IntTreeLeafT> leaves{IntTreeLeafT{1, 2, 3, 4, 5}, IntTreeLeafT{1, 2, 3}};
  List<IntVecTree> ret =
      MakeMergedTrees(TreeMerger<test::IntVecTree>{}, leaves);
  ASSERT_EQ(ret->size(), 1);

  ASSERT_TRUE(ret->at(0).Has<IntTreeInnerT>());
  const auto& [inner_data0, children0] =
      ret->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data0 == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->size() == 2));

  ASSERT_TRUE((children0->at(0).Has<IntTreeInnerT>()));
  const auto& [inner_data_left0, children_left0] =
      children0->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data_left0 == IntTreeLeafT{4, 5}));
  ASSERT_TRUE((children_left0->size() == 1));
  ASSERT_TRUE((children_left0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children_left0->at(0).Get<IntTreeLeafT>() ==
               IntTreeLeafT{1, 2, 3, 4, 5}));

  ASSERT_TRUE((children0->at(1).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children0->at(1).Get<IntTreeLeafT>() == IntTreeLeafT{1, 2, 3}));
}

TEST(IntVecTree, left_lt_right) {
  List<IntTreeLeafT> leaves{IntTreeLeafT{1, 2, 3}, IntTreeLeafT{1, 2, 3, 4, 5}};
  List<IntVecTree> ret =
      MakeMergedTrees(TreeMerger<test::IntVecTree>{}, leaves);
  ASSERT_EQ(ret->size(), 1);

  ASSERT_TRUE(ret->at(0).Has<IntTreeInnerT>());
  const auto& [inner_data0, children0] =
      ret->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data0 == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->size() == 2));

  ASSERT_TRUE((children0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children0->at(0).Get<IntTreeLeafT>() == IntTreeLeafT{1, 2, 3}));

  ASSERT_TRUE((children0->at(1).Has<IntTreeInnerT>()));
  const auto& [inner_data_right0, children_right0] =
      children0->at(1).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data_right0 == IntTreeLeafT{4, 5}));
  ASSERT_TRUE((children_right0->size() == 1));
  ASSERT_TRUE((children_right0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children_right0->at(0).Get<IntTreeLeafT>() ==
               IntTreeLeafT{1, 2, 3, 4, 5}));
}

TEST(IntVecTree, left_ne_right) {
  List<IntTreeLeafT> leaves{IntTreeLeafT{1, 2, 3, 4, 5},
                            IntTreeLeafT{1, 2, 3, 6, 7}};
  List<IntVecTree> ret =
      MakeMergedTrees(TreeMerger<test::IntVecTree>{}, leaves);
  ASSERT_EQ(ret->size(), 1);

  ASSERT_TRUE(ret->at(0).Has<IntTreeInnerT>());
  const auto& [inner_data0, children0] =
      ret->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data0 == IntTreeLeafT{1, 2, 3}));
  ASSERT_TRUE((children0->size() == 2));

  ASSERT_TRUE((children0->at(0).Has<IntTreeInnerT>()));
  const auto& [inner_data_left0, children_left0] =
      children0->at(0).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data_left0 == IntTreeLeafT{4, 5}));
  ASSERT_TRUE((children_left0->size() == 1));
  ASSERT_TRUE((children_left0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children_left0->at(0).Get<IntTreeLeafT>() ==
               IntTreeLeafT{1, 2, 3, 4, 5}));

  ASSERT_TRUE((children0->at(1).Has<IntTreeInnerT>()));
  const auto& [inner_data_right0, children_right0] =
      children0->at(1).Get<IntTreeInnerT>().tuple();
  ASSERT_TRUE((inner_data_right0 == IntTreeLeafT{6, 7}));
  ASSERT_TRUE((children_right0->size() == 1));
  ASSERT_TRUE((children_right0->at(0).Has<IntTreeLeafT>()));
  ASSERT_TRUE((children_right0->at(0).Get<IntTreeLeafT>() ==
               IntTreeLeafT{1, 2, 3, 6, 7}));
}

}  // namespace test
}  // namespace cinn::adt
