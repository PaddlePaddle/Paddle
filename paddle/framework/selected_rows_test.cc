/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/selected_rows.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

class SelectedRowsTester : public ::testing::Test {
 public:
  virtual void SetUp() override {
    std::vector<int64_t> rows{0, 4, 7};
    int64_t height = 10;
    int64_t row_numel = 100;
    selected_rows_.reset(new SelectedRows(rows, height));

    Tensor* value = selected_rows_->mutable_value();
    value->mutable_data<float>(
        make_ddim({static_cast<int64_t>(rows.size()), row_numel}), place_);
  }

 protected:
  platform::CPUPlace place_;
  std::unique_ptr<SelectedRows> selected_rows_{nullptr};
};

TEST_F(SelectedRowsTester, height) { ASSERT_EQ(selected_rows_->height(), 10); }

TEST_F(SelectedRowsTester, dims) {
  ASSERT_EQ(selected_rows_->value().dims(), make_ddim({3, 100}));
}

TEST_F(SelectedRowsTester, complete_dims) {
  ASSERT_EQ(selected_rows_->GetCompleteDims(), make_ddim({10, 100}));
}

}  // namespace framework
}  // namespace paddle
