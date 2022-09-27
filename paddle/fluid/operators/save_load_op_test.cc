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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(save);
PD_DECLARE_KERNEL(save, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(save_sr, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cast, CPU, ALL_LAYOUT);
USE_OP_ITSELF(load);
PD_DECLARE_KERNEL(load, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(load_sr, CPU, ALL_LAYOUT);

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

TEST(SaveLoadOpSelectedRows, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var = scope.Var("test_var_sr");
  auto selected_rows = var->GetMutable<phi::SelectedRows>();
  selected_rows->set_height(3);
  selected_rows->set_rows({0, 1, 2});
  auto* tensor = selected_rows->mutable_value();
  tensor->Resize({3, 10});
  int* expect = tensor->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<int>(i);
  }
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("selected_rows.save")});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var_sr"}}}, {}, attrs);
  save_op->Run(scope, place);

  auto load_var = scope.Var("out_var_sr");
  auto target = load_var->GetMutable<phi::SelectedRows>();
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var_sr"}}}, attrs);
  load_op->Run(scope, place);
  const int* actual = target->value().data<int>();
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    EXPECT_EQ(expect[i], actual[i]);
  }
  EXPECT_EQ(target->height(), 3);
  auto& rows = target->rows();
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i], static_cast<int64_t>(i));
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
