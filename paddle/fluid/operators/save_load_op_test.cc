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
#include "paddle/fluid/platform/float16.h"

USE_NO_KERNEL_OP(save);
USE_NO_KERNEL_OP(load);

TEST(SaveLoadOp, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var = scope.Var("test_var");
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({3, 10});
  paddle::framework::LoD expect_lod;
  expect_lod.resize(1);
  expect_lod[0].push_back(0);
  expect_lod[0].push_back(1);
  expect_lod[0].push_back(2);
  expect_lod[0].push_back(3);

  tensor->set_lod(expect_lod);
  int* expect = tensor->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<int>(i);
  }
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("tensor.save")});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var");
  auto target = load_var->GetMutable<paddle::framework::LoDTensor>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);
  int* actual = target->data<int>();
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    EXPECT_EQ(expect[i], actual[i]);
  }
  auto& actual_lod = target->lod();
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

TEST(SaveFP16Op, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var = scope.Var("test_var");
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({3, 10});
  paddle::framework::LoD expect_lod;
  expect_lod.resize(1);
  expect_lod[0].push_back(0);
  expect_lod[0].push_back(1);
  expect_lod[0].push_back(2);
  expect_lod[0].push_back(3);

  tensor->set_lod(expect_lod);
  float* expect = tensor->mutable_data<float>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<float>(paddle::platform::float16(i));
  }

  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("tensor.save")});
  attrs.insert({"save_as_fp16", true});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var");
  auto target = load_var->GetMutable<paddle::framework::LoDTensor>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);
  paddle::platform::float16* actual = target->data<paddle::platform::float16>();
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    EXPECT_EQ(expect[i], static_cast<float>(actual[i]));
  }
  auto& actual_lod = target->lod();
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

TEST(LoadFP16Op, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var = scope.Var("test_var");
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({3, 10});

  paddle::framework::LoD expect_lod;
  expect_lod.resize(1);
  expect_lod[0].push_back(0);
  expect_lod[0].push_back(1);
  expect_lod[0].push_back(2);
  expect_lod[0].push_back(3);

  tensor->set_lod(expect_lod);
  float* expect = tensor->mutable_data<float>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<float>(paddle::platform::float16(i));
  }

  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("tensor.save")});
  attrs.insert({"load_as_fp16", true});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var");
  load_var->GetMutable<paddle::framework::LoDTensor>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);

  auto target = load_var->Get<paddle::framework::LoDTensor>();
  paddle::platform::float16* actual = target.data<paddle::platform::float16>();
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    EXPECT_EQ(expect[i], static_cast<float>(actual[i]));
  }

  auto& actual_lod = target.lod();
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

TEST(SaveLoadSelectedRows, CPU) {
  constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";

  const std::string save_file_path = "selected_rows.save";

  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto* lt_var = scope.Var(LOOKUP_TABLE_PATH)->GetMutable<std::string>();
  lt_var->append(save_file_path);

  int64_t table_size = 100000;
  int64_t embedding_width = 8;

  auto var = scope.Var("test_var");
  auto table = var->GetMutable<paddle::framework::SelectedRows>();

  // initialize a sparse table
  table->mutable_value()->Resize(
      paddle::framework::make_ddim({table_size, embedding_width}));
  auto* expect = table->mutable_value()->mutable_data<float>(place);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      auto index = i * embedding_width + j;
      expect[index] = static_cast<float>(index);
    }
  }

  const size_t shard_num = 13;  // default value in framework
  const size_t shard_size = table_size / shard_num;

  std::vector<int64_t> ids = {1, 2, 3, 4};
  std::vector<int64_t> indexs(ids.size());
  table->InitDataShards();
  table->GetIndexsByIds(ids, &indexs, true);
  size_t id_num = ids.size();
  for (size_t i = 0; i < id_num; ++i) {
    size_t shard_id = ids[i] % shard_num;
    ASSERT_EQ(indexs[i], shard_id * shard_size);
  }

  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", save_file_path});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var");
  auto target = load_var->GetMutable<paddle::framework::SelectedRows>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);

  auto* actual = target->value().data<float>();
  for (int64_t i = 0; i < target->value().numel(); ++i) {
    EXPECT_EQ(expect[i], actual[i]);
  }

  indexs.clear();
  indexs.resize(ids.size());
  target->GetIndexsByIds(ids, &indexs, false);
  for (size_t i = 0; i < id_num; ++i) {
    size_t shard_id = ids[i] % shard_num;
    ASSERT_EQ(indexs[i], shard_id * shard_size);
  }
}
