// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/reader.h"

#include <memory>

#include "gtest/gtest.h"
#include "gtest/gtest_pred_impl.h"

class StubDecoratedReader : public paddle::framework::DecoratedReader {
 public:
  explicit StubDecoratedReader(const std::shared_ptr<ReaderBase> &reader)
      : DecoratedReader(reader) {}

  void ReadNextImpl(std::vector<paddle::framework::LoDTensor> *out) override {}
};

class StubRootReader : public paddle::framework::ReaderBase {
 public:
  explicit StubRootReader(
      const std::vector<paddle::framework::DDim> &dims,
      const std::vector<paddle::framework::proto::VarType::Type> &var_types,
      const std::vector<bool> &need_check_feed)
      : paddle::framework::ReaderBase(dims, var_types, need_check_feed) {}
  void ReadNextImpl(std::vector<paddle::framework::LoDTensor> *out) override {}
};

TEST(READER, decorate_chain) {
  paddle::framework::proto::VarType::Type dtype =
      paddle::framework::proto::VarType::FP32;
  paddle::framework::DDim dim = phi::make_ddim({5, 7});
  std::vector<paddle::framework::DDim> init_dims(4, dim);
  std::vector<paddle::framework::proto::VarType::Type> init_types(4, dtype);
  std::vector<bool> init_need_check(4, true);
  auto root =
      std::make_shared<StubRootReader>(init_dims, init_types, init_need_check);
  auto end_point1 =
      paddle::framework::MakeDecoratedReader<StubDecoratedReader>(root);
  auto end_point2 =
      paddle::framework::MakeDecoratedReader<StubDecoratedReader>(root);

  {
    auto endpoints = root->GetEndPoints();
    ASSERT_EQ(endpoints.size(), 2U);
    ASSERT_NE(endpoints.count(end_point1.get()), 0UL);
    ASSERT_NE(endpoints.count(end_point2.get()), 0UL);
  }

  {
    auto end_point3 =
        paddle::framework::MakeDecoratedReader<StubDecoratedReader>(root);
    ASSERT_EQ(root->GetEndPoints().size(), 3U);
  }
  { ASSERT_EQ(root->GetEndPoints().size(), 2U); }

  {
    ASSERT_EQ(end_point1->Shapes(), init_dims);
    ASSERT_EQ(end_point1->VarTypes(), init_types);
    ASSERT_EQ(end_point1->NeedCheckFeed(), init_need_check);
  }
}
