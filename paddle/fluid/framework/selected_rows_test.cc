/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/selected_rows.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

class SelectedRowsTester : public ::testing::Test {
 public:
  void SetUp() override {
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

TEST_F(SelectedRowsTester, SerializeAndDeseralize) {
  SelectedRows dst_tensor;
  platform::CPUDeviceContext cpu_ctx(place_);
  std::ostringstream oss;

  SerializeToStream(oss, *selected_rows_, cpu_ctx);

  std::istringstream iss(oss.str());
  DeserializeFromStream(iss, &dst_tensor, cpu_ctx);

  ASSERT_EQ(selected_rows_->rows(), dst_tensor.rows());
  ASSERT_EQ(selected_rows_->height(), dst_tensor.height());
  ASSERT_EQ(selected_rows_->value().dims(), dst_tensor.value().dims());
  ASSERT_EQ(selected_rows_->GetCompleteDims(), dst_tensor.GetCompleteDims());
}

TEST_F(SelectedRowsTester, SparseTable) {
  platform::CPUPlace cpu;
  SelectedRows table;
  // initialize a sparse table
  table.mutable_value()->Resize(framework::make_ddim({1, 100}));
  table.mutable_value()->mutable_data<float>(cpu);
  table.mutable_rows()->push_back(1);

  int64_t key = 10000;
  int64_t non_key = 999;
  framework::Tensor value;
  value.Resize(framework::make_ddim({1, 100}));
  auto ptr = value.mutable_data<float>(cpu);
  ptr[0] = static_cast<float>(10);

  ASSERT_EQ(table.rows().size(), static_cast<size_t>(1));
  ASSERT_EQ(table.HasKey(key), false);

  table.Set(key, value);

  ASSERT_EQ(table.rows().size(), static_cast<size_t>(2));
  ASSERT_EQ(table.HasKey(key), true);
  // check re-allocate
  ASSERT_EQ(table.value().dims()[0], static_cast<int64_t>(4));

  framework::Tensor get_value;
  get_value.mutable_data<float>(framework::make_ddim({2, 100}), cpu);
  std::vector<int64_t> keys({non_key, key});
  auto non_key_pairs = table.Get(keys, &get_value);

  ASSERT_EQ(get_value.data<float>()[100], static_cast<float>(10));
  ASSERT_EQ(non_key_pairs.size(), static_cast<size_t>(1));
  ASSERT_EQ(non_key_pairs[0].first, non_key);
}

}  // namespace framework
}  // namespace paddle
