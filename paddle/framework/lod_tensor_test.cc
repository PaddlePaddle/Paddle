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
#include <algorithm>
#include <memory>

namespace paddle {
namespace framework {

// TEST(LODTensor, test) {
//   LOD lod;
//   lod.push_back(std::vector<size_t>{0, 10, 20});
//   lod.push_back(std::vector<size_t>{0, 5, 10, 15, 20});
//   lod.push_back(std::vector<size_t>{0, 2, 5, 7, 10, 12, 15, 17, 20});

//   ASSERT_EQ(lod.size(), 3UL);

//   Tensor tensor;
//   tensor.Resize({20 /*batch size*/, 128 /*dim*/});
//   // malloc memory
//   platform::CPUPlace place;
//   tensor.mutable_data<float>(place);

//   LODTensor lod_tensor(lod, &tensor);

//   ASSERT_EQ(lod_tensor.NumLevels(), 3UL);
// }

class LODTensorTester : public ::testing::Test {
 public:
  virtual void SetUp() override {
    // tensor's batch_size: 30
    // 3 levels
    // 0 10 20
    // 0 5 10 15 20
    // 0 2 5 7 10 12 15 20
    LOD lod;
    lod.push_back(std::vector<size_t>{0, 10, 20});
    lod.push_back(std::vector<size_t>{0, 5, 10, 15, 20});
    lod.push_back(std::vector<size_t>{0, 2, 5, 7, 10, 12, 15, 17, 20});

    ASSERT_EQ(lod.size(), 3UL);

    tensor.Resize({20 /*batch size*/, 128 /*dim*/});
    // malloc memory
    tensor.mutable_data<float>(place);

    lod_tensor.lod_ = lod;
    lod_tensor.tensor_ = &tensor;
  }

 protected:
  platform::CPUPlace place;
  Tensor tensor;
  LODTensor lod_tensor;
};

TEST_F(LODTensorTester, NumLevels) { ASSERT_EQ(lod_tensor.NumLevels(), 3UL); }

TEST_F(LODTensorTester, NumElements) {
  ASSERT_EQ(lod_tensor.NumElements(0), 2UL);
  ASSERT_EQ(lod_tensor.NumElements(1), 4UL);
  ASSERT_EQ(lod_tensor.NumElements(2), 8UL);
}

TEST_F(LODTensorTester, SliceLevels) {
  // slice 1 level
  for (size_t level = 0; level < 3UL; ++level) {
    LODTensor new_lod_tensor = lod_tensor;
    new_lod_tensor.SliceLevels(level, level + 1);
    ASSERT_EQ(new_lod_tensor.NumLevels(), 1UL);
    ASSERT_EQ(new_lod_tensor.NumElements(0), lod_tensor.NumElements(level));
    ASSERT_EQ(new_lod_tensor.tensor_->data<float>(),
              lod_tensor.tensor_->data<float>());
  }
  // slice 2 level
  for (size_t level = 0; level < 2UL; ++level) {
    LODTensor new_lod_tensor = lod_tensor;
    new_lod_tensor.SliceLevels(level, level + 2);
    ASSERT_EQ(new_lod_tensor.NumLevels(), 2UL);
    ASSERT_EQ(new_lod_tensor.NumElements(0), lod_tensor.NumElements(level));
    ASSERT_EQ(new_lod_tensor.NumElements(1), lod_tensor.NumElements(level + 1));
    ASSERT_EQ(new_lod_tensor.tensor_->data<float>(),
              lod_tensor.tensor_->data<float>());
  }
}

TEST_F(LODTensorTester, SliceInLevel) {
  size_t level = 0;
  LODTensor new_lod_tensor = lod_tensor;
  new_lod_tensor.SliceInLevel(level, 0, 2);
  EXPECT_EQ(new_lod_tensor.NumLevels(), 3UL);
  EXPECT_EQ(new_lod_tensor.NumElements(0), 2UL);
  EXPECT_EQ(new_lod_tensor.NumElements(1), 4UL);
  EXPECT_EQ(new_lod_tensor.NumElements(2), 8UL);
  ASSERT_EQ(new_lod_tensor.tensor_->data<float>(),
            lod_tensor.tensor_->data<float>());

  level = 1;
  new_lod_tensor = lod_tensor;
  new_lod_tensor.SliceInLevel(level, 0, 2);
  ASSERT_EQ(new_lod_tensor.NumLevels(), 2UL);
  ASSERT_EQ(new_lod_tensor.NumElements(0), 2UL);
  ASSERT_EQ(new_lod_tensor.NumElements(1), 4UL);
  ASSERT_EQ(new_lod_tensor.tensor_->data<float>(),
            lod_tensor.tensor_->data<float>());
}

}  // namespace framework
}  // namespace paddle
