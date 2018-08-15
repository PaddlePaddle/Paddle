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

  int64_t table_size = 100;
  int64_t embedding_width = 8;
  // initialize a sparse table
  table.mutable_value()->Resize(
      framework::make_ddim({table_size, embedding_width}));
  auto* data = table.mutable_value()->mutable_data<float>(cpu);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      data[i * embedding_width + j] = static_cast<float>(i);
    }
  }
  ASSERT_EQ(table.AutoGrownIndex(10), 0);
  ASSERT_EQ(table.AutoGrownIndex(8), 1);
  ASSERT_EQ(table.AutoGrownIndex(8), 1);
  ASSERT_EQ(table.AutoGrownIndex(6), 2);
  ASSERT_TRUE(table.HasKey(10));
  ASSERT_TRUE(table.HasKey(8));
  ASSERT_TRUE(table.HasKey(6));
  ASSERT_EQ(table.rows().size(), 3);

  framework::Tensor ids;
  ids.Resize(framework::make_ddim({4}));
  auto* ids_data = ids.mutable_data<int64_t>(cpu);
  ids_data[0] = static_cast<int64_t>(6);
  ids_data[1] = static_cast<int64_t>(6);
  ids_data[2] = static_cast<int64_t>(8);
  ids_data[3] = static_cast<int64_t>(10);

  framework::Tensor get_value;
  auto* value_data = get_value.mutable_data<float>(
      framework::make_ddim({4, embedding_width}), cpu);
  table.Get(ids, &get_value);

  for (int j = 0; j < embedding_width; ++j) {
    ASSERT_EQ(value_data[0 * embedding_width + j], 2);
  }
  for (int j = 0; j < embedding_width; ++j) {
    ASSERT_EQ(value_data[1 * embedding_width + j], 2);
  }
  for (int j = 0; j < embedding_width; ++j) {
    ASSERT_EQ(value_data[2 * embedding_width + j], 1);
  }
  for (int j = 0; j < embedding_width; ++j) {
    ASSERT_EQ(value_data[3 * embedding_width + j], 0);
  }
}

}  // namespace framework
}  // namespace paddle
