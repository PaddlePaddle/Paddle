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

#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"

USE_NO_KERNEL_OP(save_combine);
USE_NO_KERNEL_OP(load_combine);

TEST(SaveLoadCombineOp, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  auto var1 = scope.Var("test_var1");
  auto tensor1 = var->GetMutable<paddle::framework::LoDTensor>();
  tensor1->Resize({10, 10});
  paddle::framework::LoD expect_lod1;
  expect_lod1.resize(1);
  expect_lod1[0].push_back(0);
  expect_lod1[0].push_back(1);
  expect_lod1[0].push_back(2);
  expect_lod1[0].push_back(3);

  tensor1->set_lod(expect_lod1);
  int* expect1 = tensor1->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor1->numel(); ++i) {
    expect1[i] = static_cast<int>(i);
  }
  paddle::framework::AttributeMap attrs1;
  attrs1.insert({"file_path", std::string("tensor.save")});
  attrs1.insert({"position_counter", 0});

  auto save_op1 = paddle::framework::OpRegistry::CreateOp(
      "save_combine", {{"X", {"test_var1"}}}, {}, attrs1);
  save_op1->Run(scope, place);

  auto var2 = scope.Var("test_var2");
  auto tensor2 = var->GetMutable<paddle::framework::LoDTensor>();
  tensor2->Resize({20, 20});
  paddle::framework::LoD expect_lod2;
  expect_lod2.resize(1);
  expect_lod2[0].push_back(0);
  expect_lod2[0].push_back(1);
  expect_lod2[0].push_back(2);
  expect_lod2[0].push_back(3);

  tensor2->set_lod(expect_lod2);
  int* expect2 = tensor2->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor2->numel(); ++i) {
    expect2[i] = static_cast<int>(i + 2);
  }
  paddle::framework::AttributeMap attrs2;
  attrs2.insert({"file_path", std::string("tensor.save")});
  attrs2.insert({"position_counter", 1});

  auto save_op2 = paddle::framework::OpRegistry::CreateOp(
      "save_combine", {{"X", {"test_var2"}}}, {}, attrs2);
  save_op2->Run(scope, place);

  auto load_var1 = scope.Var("out_var1");
  auto target1 = load_var1->GetMutable<paddle::framework::LoDTensor>();
  auto load_op1 = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {}, {{"Out", {"out_var1"}}}, attrs1);
  load_op1->Run(scope, place);
  int* actual1 = target1->data<int>();
  for (int64_t i = 0; i < tensor1->numel(); ++i) {
    EXPECT_EQ(expect1[i], actual1[i]);
  }
  auto& actual_lod1 = target1->lod();
  EXPECT_EQ(expect_lod1.size(), actual_lod1.size());
  for (size_t i = 0; i < expect_lod1.size(); ++i) {
    for (size_t j = 0; j < expect_lod1[i].size(); ++j) {
      EXPECT_EQ(expect_lod1[i][j], actual_lod1[i][j]);
    }
  }

  auto load_var2 = scope.Var("out_var2");
  auto target2 = load_var2->GetMutable<paddle::framework::LoDTensor>();
  auto load_op2 = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {}, {{"Out", {"out_var2"}}}, attrs2);
  load_op2->Run(scope, place);
  int* actual2 = target2->data<int>();
  for (int64_t i = 0; i < tensor2->numel(); ++i) {
    EXPECT_EQ(expect2[i], actual2[i]);
  }
  auto& actual_lod2 = target2->lod();
  EXPECT_EQ(expect_lod2.size(), actual_lod2.size());
  for (size_t i = 0; i < expect_lod2.size(); ++i) {
    for (size_t j = 0; j < expect_lod2[i].size(); ++j) {
      EXPECT_EQ(expect_lod2[i][j], actual_lod2[i][j]);
    }
  }
}
