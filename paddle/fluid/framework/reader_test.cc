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

class StubDecoratedReader : public paddle::framework::DecoratedReader {
 public:
  explicit StubDecoratedReader(const std::shared_ptr<ReaderBase> &reader)
      : DecoratedReader(reader) {}

  void ReadNextImpl(std::vector<paddle::framework::LoDTensor> *out) override {}
};

class StubRootReader : public paddle::framework::ReaderBase {
 public:
  void ReadNextImpl(std::vector<paddle::framework::LoDTensor> *out) override {}
};

TEST(READER, decorate_chain) {
  auto root = std::make_shared<StubRootReader>();
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
}
