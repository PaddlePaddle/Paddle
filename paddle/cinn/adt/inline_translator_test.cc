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

#include "paddle/cinn/adt/inline_translator.h"

#include <string>

#include "gtest/gtest.h"

namespace cinn::adt {

template <>
struct InlineTranslatorTrait<List> final {
  template <typename T>
  static List<T> GetTreeInnerNodeChildren(const List<T>& list) {
    return list;
  }

  template <typename SrcTreeT, typename DstTreeT>
  static List<DstTreeT> ConvertMap(const List<SrcTreeT>& src_map,
                                   const List<DstTreeT>& dst_children) {
    return dst_children;
  }
};

namespace test {

using SrcLeaf = Store<std::string, List<Load<std::string>>>;
using DstLeaf = Store<std::string, Tree<List, Load<std::string>>>;
using SrcTree = Tree<List, SrcLeaf>;
using DstTree = Tree<List, DstLeaf>;

// (Tree List (Store string (List (Load string)))) ->
// (Tree List (Store string (Tree List (Load string))))

// Src:
// c = [a, b];
// d = [c, e];
// f = [d, c];
// Dst:
// f = [[[a, b], e], [a, b]];
TEST(InlineTranslator, Naive) {
  List<Tree<List, SrcLeaf>> src_op_calls{};
  src_op_calls->emplace_back(SrcLeaf{
      "c",
      List<Load<std::string>>{Load<std::string>{"a"}, Load<std::string>{"b"}}});
  src_op_calls->emplace_back(SrcLeaf{
      "d",
      List<Load<std::string>>{Load<std::string>{"c"}, Load<std::string>{"e"}}});
  src_op_calls->emplace_back(SrcLeaf{
      "f",
      List<Load<std::string>>{Load<std::string>{"d"}, Load<std::string>{"c"}}});
  SrcTree src_tree{src_op_calls};
  const DstTree& dst_tree =
      InlineTranslator<List, List, std::string>::Call(src_tree);
  ASSERT_TRUE((dst_tree.Has<List<Tree<List, DstLeaf>>>()));
  const auto& dst_level0_leaves = dst_tree.Get<List<Tree<List, DstLeaf>>>();
  ASSERT_EQ(dst_level0_leaves->size(), 1);
  const auto& dst_level0_leaf = dst_level0_leaves->at(0);
  ASSERT_TRUE((dst_level0_leaf.Has<DstLeaf>()));
  const auto& [f, f_tree] = dst_level0_leaf.Get<DstLeaf>().tuple();
  ASSERT_EQ(f, "f");
  using NestedList = Tree<List, Load<std::string>>;
  ASSERT_TRUE((f_tree.Has<List<NestedList>>()));
  ASSERT_EQ((f_tree.Get<List<NestedList>>()->size()), 2);
  {
    // [[a, b], e]
    NestedList d = f_tree.Get<List<NestedList>>()->at(0);
    ASSERT_TRUE((d.Has<List<NestedList>>()));
    ASSERT_EQ((d.Get<List<NestedList>>()->size()), 2);
    {
      // [a, b]
      NestedList c = d.Get<List<NestedList>>()->at(0);
      ASSERT_TRUE((c.Has<List<NestedList>>()));
      ASSERT_EQ((c.Get<List<NestedList>>()->size()), 2);
      {
        NestedList a = c.Get<List<NestedList>>()->at(0);
        ASSERT_TRUE((a.Has<Load<std::string>>()));
        const auto& [a_string] = a.Get<Load<std::string>>().tuple();
        ASSERT_EQ(a_string, "a");
      }
      {
        NestedList b = c.Get<List<NestedList>>()->at(1);
        ASSERT_TRUE((b.Has<Load<std::string>>()));
        const auto& [b_string] = b.Get<Load<std::string>>().tuple();
        ASSERT_EQ(b_string, "b");
      }
    }
    {
      NestedList e = d.Get<List<NestedList>>()->at(1);
      ASSERT_TRUE((e.Has<Load<std::string>>()));
      const auto& [e_string] = e.Get<Load<std::string>>().tuple();
      ASSERT_EQ(e_string, "e");
    }
  }
  {
    // [a, b]
    NestedList c = f_tree.Get<List<NestedList>>()->at(1);
    ASSERT_TRUE((c.Has<List<NestedList>>()));
    ASSERT_EQ((c.Get<List<NestedList>>()->size()), 2);
    {
      NestedList a = c.Get<List<NestedList>>()->at(0);
      ASSERT_TRUE((a.Has<Load<std::string>>()));
      const auto& [a_string] = a.Get<Load<std::string>>().tuple();
      ASSERT_EQ(a_string, "a");
    }
    {
      NestedList b = c.Get<List<NestedList>>()->at(1);
      ASSERT_TRUE((b.Has<Load<std::string>>()));
      const auto& [b_string] = b.Get<Load<std::string>>().tuple();
      ASSERT_EQ(b_string, "b");
    }
  }
}

}  // namespace test

}  // namespace cinn::adt
