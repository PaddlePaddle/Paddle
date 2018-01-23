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

#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"

USE_NO_KERNEL_OP(save_combine);
USE_NO_KERNEL_OP(load_combine);

int* create_and_run_save_combine_op(
    int x, int y, const std::vector<int>& lod_info, std::string var_name,
    int pos_counter, std::string filename, paddle::framework::Scope& scope,
    paddle::platform::CPUPlace& place, paddle::framework::LoD& expect_lod) {
  auto var = scope.Var(var_name);
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({x, y});
  expect_lod.resize(1);
  for (size_t i = 0; i < lod_info.size(); i++) {
    expect_lod[0].push_back(lod_info[i]);
  }

  tensor->set_lod(expect_lod);
  int* expect = tensor->mutable_data<int>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<int>(i);
  }
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});
  attrs.insert({"position_counter", static_cast<int>(pos_counter)});
  auto save_combine_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine", {{"X", {var_name}}}, {}, attrs);
  save_combine_op->Run(scope, place);
  return expect;
}

int* create_and_run_load_combine_op(std::string var_name, int pos_counter,
                                    std::string filename,
                                    paddle::framework::Scope& scope,
                                    paddle::platform::CPUPlace& place,
                                    paddle::framework::LoD& actual_lod) {
  auto load_var = scope.Var(var_name);
  auto target = load_var->GetMutable<paddle::framework::LoDTensor>();
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});
  attrs.insert({"position_counter", static_cast<int>(pos_counter)});
  auto load_combine_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine", {}, {{"Out", {var_name}}}, attrs);
  load_combine_op->Run(scope, place);
  int* actual = target->data<int>();
  actual_lod = target->lod();
  return actual;
}

void check_values(int* expect, int* actual, paddle::framework::LoD expect_lod,
                  paddle::framework::LoD actual_lod, const int& numel) {
  for (int64_t i = 0; i < numel; ++i) {
    EXPECT_EQ(expect[i], actual[i]);
  }
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

// Here, we create 4 LoDTensors and use save_combine_op to first save these
// in a single file. Then, we use load_combine_op to load these sequentially
TEST(SaveLoadCombineOp, CPU) {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;

  std::vector<int> lod1 = {0, 1, 2, 3};
  int numel1 = 100;
  paddle::framework::LoD expect_lod1;
  int* expect1 = create_and_run_save_combine_op(
      10, 10, lod1, "test_var1", 0, "tensor.save", scope, place, expect_lod1);

  std::vector<int> lod2 = {0, 2, 5};
  int numel2 = 200;
  paddle::framework::LoD expect_lod2;
  int* expect2 = create_and_run_save_combine_op(
      10, 20, lod2, "test_var2", 1, "tensor.save", scope, place, expect_lod2);

  std::vector<int> lod3 = {0, 2, 3};
  int numel3 = 4000;
  paddle::framework::LoD expect_lod3;
  int* expect3 = create_and_run_save_combine_op(
      200, 20, lod3, "test_var3", 2, "tensor.save", scope, place, expect_lod3);

  std::vector<int> lod4 = {0, 1};
  int numel4 = 1000;
  paddle::framework::LoD expect_lod4;
  int* expect4 = create_and_run_save_combine_op(
      50, 20, lod4, "test_var4", 3, "tensor.save", scope, place, expect_lod4);

  paddle::framework::LoD actual_lod1, actual_lod2, actual_lod3, actual_lod4;
  int* actual1 = create_and_run_load_combine_op("out_var1", 0, "tensor.save",
                                                scope, place, actual_lod1);
  int* actual2 = create_and_run_load_combine_op("out_var2", 1, "tensor.save",
                                                scope, place, actual_lod2);
  int* actual3 = create_and_run_load_combine_op("out_var3", 2, "tensor.save",
                                                scope, place, actual_lod3);
  int* actual4 = create_and_run_load_combine_op("out_var4", 3, "tensor.save",
                                                scope, place, actual_lod4);

  check_values(expect1, actual1, expect_lod1, actual_lod1, numel1);
  check_values(expect2, actual2, expect_lod2, actual_lod2, numel2);
  check_values(expect3, actual3, expect_lod3, actual_lod3, numel3);
  check_values(expect4, actual4, expect_lod4, actual_lod4, numel4);
}
