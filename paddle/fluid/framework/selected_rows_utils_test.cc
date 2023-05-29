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

#include "paddle/fluid/framework/selected_rows_utils.h"

#include <time.h>

#include <thread>  // NOLINT

#include "gtest/gtest.h"

namespace paddle {
namespace framework {

class SelectedRowsTester : public ::testing::Test {
 public:
  void SetUp() override {
    std::vector<int64_t> rows{0, 4, 7};
    int64_t height = 10;
    int64_t row_numel = 100;
    selected_rows_.reset(new phi::SelectedRows(rows, height));

    phi::DenseTensor* value = selected_rows_->mutable_value();
    auto* data = value->mutable_data<float>(
        phi::make_ddim({static_cast<int64_t>(rows.size()), row_numel}), place_);
    for (int64_t i = 0; i < value->numel(); ++i) {
      data[i] = static_cast<float>(i);
    }
  }

 protected:
  platform::CPUPlace place_;
  std::unique_ptr<phi::SelectedRows> selected_rows_{nullptr};
};

TEST_F(SelectedRowsTester, height) { ASSERT_EQ(selected_rows_->height(), 10); }

TEST_F(SelectedRowsTester, dims) {
  ASSERT_EQ(selected_rows_->value().dims(), phi::make_ddim({3, 100}));
}

TEST_F(SelectedRowsTester, complete_dims) {
  ASSERT_EQ(selected_rows_->GetCompleteDims(), phi::make_ddim({10, 100}));
}

TEST_F(SelectedRowsTester, SerializeAndDeseralize) {
  phi::SelectedRows dst_tensor;
  phi::CPUContext cpu_ctx(place_);
  std::ostringstream oss;

  SerializeToStream(oss, *selected_rows_, cpu_ctx);

  std::istringstream iss(oss.str());
  DeserializeFromStream(iss, &dst_tensor, cpu_ctx);

  ASSERT_EQ(selected_rows_->rows(), dst_tensor.rows());
  ASSERT_EQ(selected_rows_->height(), dst_tensor.height());
  ASSERT_EQ(selected_rows_->value().dims(), dst_tensor.value().dims());
  ASSERT_EQ(selected_rows_->GetCompleteDims(), dst_tensor.GetCompleteDims());
  auto* dst_data = dst_tensor.value().data<float>();
  for (int64_t i = 0; i < dst_tensor.value().numel(); ++i) {
    ASSERT_EQ(dst_data[i], static_cast<float>(i));
  }
}

TEST(SelectedRows, SparseTable) {
  platform::CPUPlace cpu;
  phi::SelectedRows table;

  int64_t table_size = 100;
  int64_t embedding_width = 8;
  // initialize a sparse table
  table.mutable_value()->Resize(phi::make_ddim({table_size, embedding_width}));
  auto* data = table.mutable_value()->mutable_data<float>(cpu);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      data[i * embedding_width + j] = static_cast<float>(i);
    }
  }
  ASSERT_EQ(table.AutoGrownIndex(10, true, false), 0);
  ASSERT_EQ(table.AutoGrownIndex(8, true, false), 1);
  ASSERT_EQ(table.AutoGrownIndex(8, true, false), 1);
  ASSERT_EQ(table.AutoGrownIndex(6, true, false), 2);
  for (int64_t i = 11; i < 20; i++) {
    ASSERT_EQ(table.AutoGrownIndex(i, true, true), -1);
    ASSERT_TRUE(!table.HasKey(i));
  }
  ASSERT_TRUE(table.HasKey(10));
  ASSERT_TRUE(table.HasKey(8));
  ASSERT_TRUE(table.HasKey(6));
  ASSERT_EQ(table.rows().size(), 3UL);

  phi::DenseTensor ids;
  ids.Resize(phi::make_ddim({4}));
  auto* ids_data = ids.mutable_data<int64_t>(cpu);
  ids_data[0] = static_cast<int64_t>(6);
  ids_data[1] = static_cast<int64_t>(6);
  ids_data[2] = static_cast<int64_t>(8);
  ids_data[3] = static_cast<int64_t>(10);

  phi::DenseTensor get_value;
  auto* value_data =
      get_value.mutable_data<float>(phi::make_ddim({4, embedding_width}), cpu);
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

void f1(phi::SelectedRows* table, int table_size) {
  for (int i = 1000000; i > 0; --i) {
    auto id = i % table_size;
    int64_t index1 = table->AutoGrownIndex(id, true);
    int64_t index2 = table->AutoGrownIndex(id, false);
    int64_t index3 = table->AutoGrownIndex(id, true);
    ASSERT_EQ(index1, index2);
    ASSERT_EQ(index2, index3);
  }
}

void f2(phi::SelectedRows* table, int table_size) {
  for (int i = 0; i < 1000000; ++i) {
    auto id = i % table_size;
    int64_t index1 = table->AutoGrownIndex(id, true);
    int64_t index2 = table->AutoGrownIndex(id, false);
    int64_t index3 = table->AutoGrownIndex(id, true);
    ASSERT_EQ(index1, index2);
    ASSERT_EQ(index2, index3);
  }
}

void f3(phi::SelectedRows* table, int table_size) {
  clock_t t1 = clock();
  for (int i = 100000; i > 0; --i) {
    auto id1 = table->AutoGrownIndex(i % table_size, true);
    auto id2 = table->Index(i % table_size);
    ASSERT_EQ(id1, id2);
  }
  clock_t t2 = clock();
  std::cout << "f3 run time:" << t2 - t1 << std::endl;
}

void f4(phi::SelectedRows* table, int table_size) {
  clock_t t1 = clock();
  for (int i = 0; i < 100000; ++i) {
    auto id1 = table->AutoGrownIndex(i % table_size, true);
    auto id2 = table->Index(i % table_size);
    ASSERT_EQ(id1, id2);
  }
  clock_t t2 = clock();
  std::cout << "f4 run time:" << t2 - t1 << std::endl;
}

TEST(SelectedRows, MultiThreadAutoIndex) {
  platform::CPUPlace cpu;
  phi::SelectedRows table;

  int64_t table_size = 100000;
  int64_t embedding_width = 8;
  // initialize a sparse table
  table.mutable_value()->Resize(phi::make_ddim({table_size, embedding_width}));
  auto* data = table.mutable_value()->mutable_data<float>(cpu);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      data[i * embedding_width + j] = static_cast<float>(i);
    }
  }

  std::thread t1(f1, &table, table_size);
  std::thread t11(f1, &table, table_size);
  std::thread t2(f2, &table, table_size);
  std::thread t22(f2, &table, table_size);
  t1.join();
  t11.join();
  t2.join();
  t22.join();
  std::thread t3(f3, &table, table_size);
  std::thread t4(f4, &table, table_size);
  t3.join();
  t4.join();
}

}  // namespace framework
}  // namespace paddle
