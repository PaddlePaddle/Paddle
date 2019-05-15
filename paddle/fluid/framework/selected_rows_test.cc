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

#include <time.h>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/selected_rows.h"

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
    auto* data = value->mutable_data<float>(
        make_ddim({static_cast<int64_t>(rows.size()), row_numel}), place_);
    for (int64_t i = 0; i < value->numel(); ++i) {
      data[i] = static_cast<float>(i);
    }
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
  SelectedRows dst_selected_rows;
  platform::CPUDeviceContext cpu_ctx(place_);
  std::ostringstream oss;

  SerializeToStream(oss, *selected_rows_, cpu_ctx);

  std::istringstream iss(oss.str());
  DeserializeFromStream(iss, &dst_selected_rows, cpu_ctx);

  ASSERT_EQ(selected_rows_->rows(), dst_selected_rows.rows());
  ASSERT_EQ(selected_rows_->height(), dst_selected_rows.height());
  ASSERT_EQ(selected_rows_->value().dims(), dst_selected_rows.value().dims());
  ASSERT_EQ(selected_rows_->GetCompleteDims(),
            dst_selected_rows.GetCompleteDims());
  auto* dst_data = dst_selected_rows.value().data<float>();
  for (int64_t i = 0; i < dst_selected_rows.value().numel(); ++i) {
    ASSERT_EQ(dst_data[i], static_cast<float>(i));
  }
}

TEST(SelectedRows, GetIndexsByIds) {
  platform::CPUPlace cpu;
  SelectedRows table;

  int64_t table_size = 100000;
  int64_t embedding_width = 8;
  // initialize a sparse table
  table.mutable_value()->Resize(
      framework::make_ddim({table_size, embedding_width}));
  table.mutable_value()->mutable_data<float>(cpu);

  size_t shard_num = 13;  // default value in framework
  size_t shard_size = table_size / shard_num;

  std::vector<int64_t> ids = {1, 2, 3, 4};
  std::vector<int64_t> indexs(ids.size());
  table.InitDataShards();
  table.GetIndexsByIds(ids, &indexs, true);
  size_t id_num = ids.size();
  for (size_t i = 0; i < id_num; ++i) {
    size_t shard_id = ids[i] % shard_num;
    ASSERT_EQ(indexs[i], shard_id * shard_size);
  }

  // test serialize and deserialize
  table.SyncBeforeSave();
  std::unordered_map<int64_t, int64_t> id_to_offset;
  for (size_t i = 0; i < ids.size(); ++i) {
    auto& rows = table.rows();
    id_to_offset[rows[i * 2]] = rows[i * 2 + 1];
  }
  for (size_t i = 0; i < ids.size(); ++i) {
    ASSERT_EQ(id_to_offset[ids[i]], indexs[i]);
  }

  platform::CPUPlace place_;
  SelectedRows dst_selected_rows;
  platform::CPUDeviceContext cpu_ctx(place_);
  std::ostringstream oss;

  SerializeToStream(oss, table, cpu_ctx);

  std::istringstream iss(oss.str());
  DeserializeFromStream(iss, &dst_selected_rows, cpu_ctx);
  dst_selected_rows.SyncAfterLoad();

  indexs.clear();
  indexs.resize(ids.size());
  for (auto i = 0; i < indexs.size(); ++i) {
    indexs[i] = 0;
  }
  dst_selected_rows.GetIndexsByIds(ids, &indexs, false);
  for (size_t i = 0; i < id_num; ++i) {
    size_t shard_id = ids[i] % shard_num;
    ASSERT_EQ(indexs[i], shard_id * shard_size);
  }
}

TEST(SelectedRows, Get) {
  platform::CPUPlace cpu;
  SelectedRows table;

  int64_t table_size = 100000;
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

  table.InitDataShards();

  size_t shard_num = 13;  // default value in framework
  size_t shard_size = table_size / shard_num;

  int64_t id_num = 10;
  Tensor ids_t;
  Tensor out_t;

  ids_t.Resize(framework::make_ddim({id_num, 1}));
  auto* ids_data = ids_t.mutable_data<int64_t>(cpu);
  int64_t ids_num = ids_t.numel();
  for (int i = 0; i < ids_num; ++i) {
    ids_data[i] = i;
  }

  auto* out_data = out_t.Resize(framework::make_ddim({id_num, embedding_width}))
                       .mutable_data<float>(cpu);
  table.Get(ids_t, &out_t, true, false);

  for (int i = 0; i < id_num; ++i) {
    for (int j = 0; j < embedding_width; ++j) {
      size_t shard_id = ids_data[i] % shard_num;
      size_t offset = shard_id * shard_size;
      float out_val = out_data[i * embedding_width + j];
      ASSERT_EQ(out_val, offset);
    }
  }
}

}  // namespace framework
}  // namespace paddle
