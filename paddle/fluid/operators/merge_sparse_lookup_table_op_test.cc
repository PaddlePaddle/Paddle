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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/float16.h"

using Scope = paddle::framework::Scope;
using SelectedRows = paddle::framework::SelectedRows;

USE_CPU_ONLY_OP(merge_sparse_lookup_table);

namespace paddle {
namespace framework {

void InitTable(int64_t table_size, int64_t embedding_width,
               const std::vector<int64_t>& ids, std::vector<int64_t>* indexs,
               SelectedRows* table) {
  paddle::platform::CPUPlace place;

  // initialize table1
  table->mutable_value()->Resize(
      paddle::framework::make_ddim({table_size, embedding_width}));
  auto* data = table->mutable_value()->mutable_data<float>(place);
  memset(data, 0, sizeof(float) * table->value().numel());

  table->InitDataShards();
  table->set_height(table_size);
  table->GetIndexsByIds(ids, indexs, true);
  table->SyncBeforeSave();

  // set value in tensor
  for (size_t i = 0; i < ids.size(); ++i) {
    auto index = (*indexs)[i];
    auto start = data + index * embedding_width;
    for (size_t j = 0; j < embedding_width; ++j) {
      *(start + j) = static_cast<float>(ids[i]);
    }
  }
}

TEST(MergeSparseLookupTableOp, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  int64_t table_size = 100;
  int64_t embedding_width = 8;

  auto* table1 = scope.Var("x1")->GetMutable<SelectedRows>();
  auto* table2 = scope.Var("x2")->GetMutable<SelectedRows>();

  auto* out_table = scope.Var("out")->GetMutable<SelectedRows>();

  const size_t shard_num = 13;  // default value in framework
  const size_t shard_size = table_size / shard_num;

  std::vector<int64_t> ids1 = {1, 2, 3, 4};
  std::vector<int64_t> indexs1(ids1.size());

  InitTable(table_size, embedding_width, ids1, &indexs1, table1);

  std::vector<int64_t> ids2 = {5, 6, 7, 8};
  std::vector<int64_t> indexs2(ids2.size());

  InitTable(table_size, embedding_width, ids2, &indexs2, table2);

  paddle::framework::AttributeMap attrs;
  auto merge_op = paddle::framework::OpRegistry::CreateOp(
      "merge_sparse_lookup_table", {{"X", {"x1", "x2"}}}, {{"Out", {"out"}}},
      attrs);
  merge_op->Run(scope, place);

  auto* out_data = out_table->value().data<float>();

  auto& dims = out_table->value().dims();

  auto& out_id_to_index = out_table->GetIdToIndex();
  for (auto& iter : out_id_to_index) {
    float id = static_cast<float>(iter.first);
    for (size_t i = 0; i < embedding_width; ++i) {
      ASSERT_EQ(*(out_data + iter.second * dims[1] + i), id);
    }
  }
}

}  // namespace framework
}  // namespace paddle
