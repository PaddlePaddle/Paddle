/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/framework/lod_tensor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

namespace paddle {
namespace framework {

class LODTensorTester : public ::testing::Test {
 public:
  virtual void SetUp() override {
    lod_tensor.reset(new LODTensor);
    // tensor's batch_size: 30
    // 3 levels
    // 0 10 20
    // 0 5 10 15 20
    // 0 2 5 7 10 12 15 20
    auto lod =
        std::make_shared<LODTensor::lod_t>(std::vector<std::vector<size_t>>{
            {0, 10, 20}, {0, 5, 10, 15, 20}, {0, 2, 5, 7, 10, 12, 15, 17, 20}});

    auto tensor = std::make_shared<Tensor>();
    tensor->Resize({20 /*batch size*/, 128 /*dim*/});
    // malloc memory
    tensor->mutable_data<float>(place);

    lod_tensor->Reset(tensor, lod);
  }

 protected:
  std::unique_ptr<LODTensor> lod_tensor;
  platform::CPUPlace place;
};

TEST_F(LODTensorTester, Levels) { ASSERT_EQ(lod_tensor->Levels(), 3UL); }

TEST_F(LODTensorTester, Elements) {
  ASSERT_EQ(lod_tensor->Elements(0), 2UL);
  ASSERT_EQ(lod_tensor->Elements(1), 4UL);
  ASSERT_EQ(lod_tensor->Elements(2), 8UL);
}

TEST_F(LODTensorTester, SliceShared_Level) {
  // slice 1 level
  for (int level = 0; level < 3; level++) {
    auto new_lod_tensor = lod_tensor->SliceShared(level, level + 1);
    ASSERT_EQ(new_lod_tensor.Levels(), 1UL);
    ASSERT_EQ(new_lod_tensor.Elements(0UL), lod_tensor->Elements(level));
    ASSERT_EQ(new_lod_tensor.tensor(), lod_tensor->tensor());
  }
  // slice 2 level
  for (int level = 0; level < 2; level++) {
    auto new_lod_tensor = lod_tensor->SliceShared(level, level + 2);
    ASSERT_EQ(new_lod_tensor.Levels(), 2UL);
    ASSERT_EQ(new_lod_tensor.Elements(0), lod_tensor->Elements(level));
    ASSERT_EQ(new_lod_tensor.Elements(1), lod_tensor->Elements(level + 1));
    ASSERT_EQ(new_lod_tensor.tensor(), lod_tensor->tensor());
  }
}

TEST_F(LODTensorTester, SliceCopied_Level) {
  // slice 1 level
  for (int level = 0; level < 3; level++) {
    auto new_lod_tensor =
        lod_tensor->SliceCopied<float>(level, level + 1, place);
    ASSERT_EQ(new_lod_tensor.Levels(), 1UL);
    ASSERT_EQ(new_lod_tensor.Elements(0UL), lod_tensor->Elements(level));
    // ASSERT_EQ(new_lod_tensor.tensor(), lod_tensor->tensor());
    // TODO(superjom) add tensor comparation here.
  }
  // slice 2 level
  for (int level = 0; level < 2; level++) {
    auto new_lod_tensor =
        lod_tensor->SliceCopied<float>(level, level + 2, place);
    ASSERT_EQ(new_lod_tensor.Levels(), 2UL);
    ASSERT_EQ(new_lod_tensor.Elements(0), lod_tensor->Elements(level));
    ASSERT_EQ(new_lod_tensor.Elements(1), lod_tensor->Elements(level + 1));
    // ASSERT_EQ(new_lod_tensor.tensor(), lod_tensor->tensor());
    // TODO(superjom) add tensor comparation here.
  }
}

TEST_F(LODTensorTester, SliceShared_Element) {
  size_t level = 0;
  auto new_lod_tensor = lod_tensor->SliceShared<float>(level, 0, 2);
  ASSERT_EQ(new_lod_tensor.Levels(), 3UL);
  ASSERT_EQ(new_lod_tensor.Elements(0), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(1), 4UL);
  ASSERT_EQ(new_lod_tensor.Elements(2), 8UL);
  ASSERT_EQ(new_lod_tensor.raw_tensor(), lod_tensor->raw_tensor());

  level = 1;
  new_lod_tensor = lod_tensor->SliceShared<float>(level, 0, 2);
  ASSERT_EQ(new_lod_tensor.Levels(), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(0), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(1), 4UL);
  ASSERT_EQ(new_lod_tensor.raw_tensor(), lod_tensor->raw_tensor());
}

TEST_F(LODTensorTester, SliceCopied_Element) {
  size_t level = 0;
  auto new_lod_tensor = lod_tensor->SliceCopied<float>(level, 0, 2, place);
  ASSERT_EQ(new_lod_tensor.Levels(), 3UL);
  ASSERT_EQ(new_lod_tensor.Elements(0), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(1), 4UL);
  ASSERT_EQ(new_lod_tensor.Elements(2), 8UL);
  ASSERT_NE(new_lod_tensor.raw_tensor(), lod_tensor->raw_tensor());

  level = 1;
  new_lod_tensor = lod_tensor->SliceCopied<float>(level, 0, 2, place);
  ASSERT_EQ(new_lod_tensor.Levels(), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(0), 2UL);
  ASSERT_EQ(new_lod_tensor.Elements(1), 4UL);
  ASSERT_NE(new_lod_tensor.raw_tensor(), lod_tensor->raw_tensor());
  // TODO(superjom) compare the content of these tensors
}

TEST_F(LODTensorTester, ShareLOD) {
  LODTensor new_lod_tensor;
  new_lod_tensor.ShareLOD(*lod_tensor);
  ASSERT_EQ(new_lod_tensor.lod(), lod_tensor->lod());
}

TEST_F(LODTensorTester, CopyLOD) {
  LODTensor new_lod_tensor;
  new_lod_tensor.CopyLOD(*lod_tensor);
  ASSERT_NE(new_lod_tensor.lod(), lod_tensor->lod());
}

}  // namespace framework
}  // namespace paddle
