// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/xpu/xpu_info.h"  // Include XPU initialization headers
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

template <typename Place, typename T>
T* CreateForSaveCombineOp(int x,
                          int y,
                          const std::vector<int>& lod_info,
                          std::string var_name,
                          const Place& place,
                          paddle::framework::Scope* scope,
                          phi::LegacyLoD* expect_lod) {
  phi::CPUPlace cpu_place;
  std::vector<T> ground_truth_cpu(x * y);
  for (int i = 0; i < x * y; ++i) {
    ground_truth_cpu[i] = static_cast<T>(i);
  }

  auto var = scope->Var(var_name);
  auto tensor = var->GetMutable<phi::DenseTensor>();
  tensor->Resize({x, y});
  expect_lod->resize(1);
  for (auto item : lod_info) {
    (*expect_lod)[0].push_back(item);
  }
  tensor->set_lod(*expect_lod);
  T* expect = tensor->mutable_data<T>(place);
  paddle::memory::Copy(place,
                       expect,
                       cpu_place,
                       ground_truth_cpu.data(),
                       sizeof(T) * ground_truth_cpu.size());
  return expect;
}

phi::DenseTensor* GeneratePlaceholderBeforeLoad(
    const std::string& out_var_name, paddle::framework::Scope* scope) {
  auto load_var = scope->Var(out_var_name);
  auto target = load_var->GetMutable<phi::DenseTensor>();
  return target;
}

template <typename T>
T* GetValuesAfterLoadCombineOp(phi::DenseTensor* target,
                               const paddle::framework::Scope& scope,
                               phi::LegacyLoD* actual_lod) {
  T* actual = target->data<T>();
  *actual_lod = target->lod();
  return actual;
}

template <typename T, typename U>
void CheckValues(T* expect,
                 U* actual,
                 const phi::LegacyLoD& expect_lod,
                 const phi::LegacyLoD& actual_lod,
                 const int& numel) {
  for (int i = 0; i < numel; ++i) {
    EXPECT_EQ(expect[i], static_cast<T>(actual[i]));
  }
  EXPECT_EQ(expect_lod.size(), actual_lod.size());
  for (size_t i = 0; i < expect_lod.size(); ++i) {  // NOLINT
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      EXPECT_EQ(expect_lod[i][j], actual_lod[i][j]);
    }
  }
}

template <typename Place, typename T, typename U>
int SaveLoadCombineOpTest(Place place) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  std::vector<int> lod1 = {0, 1, 2, 3, 10};
  int numel1 = 100;
  phi::LegacyLoD expect_lod1;
  T* expect1 = CreateForSaveCombineOp<Place, T>(
      10, 10, lod1, "test_var1", place, &scope, &expect_lod1);

  std::vector<int> lod2 = {0, 2, 5, 10};
  int numel2 = 200;
  phi::LegacyLoD expect_lod2;
  T* expect2 = CreateForSaveCombineOp<Place, T>(
      10, 20, lod2, "test_var2", place, &scope, &expect_lod2);

  std::vector<int> lod3 = {0, 2, 3, 20};
  int numel3 = 4000;
  phi::LegacyLoD expect_lod3;
  T* expect3 = CreateForSaveCombineOp<Place, T>(
      20, 200, lod3, "test_var3", place, &scope, &expect_lod3);

  std::vector<int> lod4 = {0, 1, 20};
  int numel4 = 1000;
  phi::LegacyLoD expect_lod4;
  T* expect4 = CreateForSaveCombineOp<Place, T>(
      20, 50, lod4, "test_var4", place, &scope, &expect_lod4);

  std::string filename = "check_tensor.ls";
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string(filename)});

  auto save_combine_op = paddle::framework::OpRegistry::CreateOp(
      "save_combine",
      {{"X", {"test_var1", "test_var2", "test_var3", "test_var4"}}},
      {},
      attrs);
  save_combine_op->Run(scope, place);

  auto target1 = GeneratePlaceholderBeforeLoad("out_var1", &scope);
  auto target2 = GeneratePlaceholderBeforeLoad("out_var2", &scope);
  auto target3 = GeneratePlaceholderBeforeLoad("out_var3", &scope);
  auto target4 = GeneratePlaceholderBeforeLoad("out_var4", &scope);

  auto load_combine_op = paddle::framework::OpRegistry::CreateOp(
      "load_combine",
      {},
      {{"Out", {"out_var1", "out_var2", "out_var3", "out_var4"}}},
      attrs);
  load_combine_op->Run(scope, place);

  phi::LegacyLoD actual_lod1, actual_lod2, actual_lod3, actual_lod4;
  U* actual1 = GetValuesAfterLoadCombineOp<U>(target1, scope, &actual_lod1);
  U* actual2 = GetValuesAfterLoadCombineOp<U>(target2, scope, &actual_lod2);
  U* actual3 = GetValuesAfterLoadCombineOp<U>(target3, scope, &actual_lod3);
  U* actual4 = GetValuesAfterLoadCombineOp<U>(target4, scope, &actual_lod4);

  std::vector<T> expect1_cpu(numel1);
  paddle::memory::Copy(
      cpu_place, expect1_cpu.data(), place, expect1, sizeof(T) * numel1);
  std::vector<T> expect2_cpu(numel2);
  paddle::memory::Copy(
      cpu_place, expect2_cpu.data(), place, expect2, sizeof(T) * numel2);
  std::vector<T> expect3_cpu(numel3);
  paddle::memory::Copy(
      cpu_place, expect3_cpu.data(), place, expect3, sizeof(T) * numel3);
  std::vector<T> expect4_cpu(numel4);
  paddle::memory::Copy(
      cpu_place, expect4_cpu.data(), place, expect4, sizeof(T) * numel4);

  std::vector<U> actual1_cpu(numel1);
  paddle::memory::Copy(
      cpu_place, actual1_cpu.data(), place, actual1, sizeof(U) * numel1);
  std::vector<U> actual2_cpu(numel2);
  paddle::memory::Copy(
      cpu_place, actual2_cpu.data(), place, actual2, sizeof(U) * numel2);
  std::vector<U> actual3_cpu(numel3);
  paddle::memory::Copy(
      cpu_place, actual3_cpu.data(), place, actual3, sizeof(U) * numel3);
  std::vector<U> actual4_cpu(numel4);
  paddle::memory::Copy(
      cpu_place, actual4_cpu.data(), place, actual4, sizeof(U) * numel4);

  CheckValues(
      expect1_cpu.data(), actual1_cpu.data(), expect_lod1, actual_lod1, numel1);
  CheckValues(
      expect2_cpu.data(), actual2_cpu.data(), expect_lod2, actual_lod2, numel2);
  CheckValues(
      expect3_cpu.data(), actual3_cpu.data(), expect_lod3, actual_lod3, numel3);
  CheckValues(
      expect4_cpu.data(), actual4_cpu.data(), expect_lod4, actual_lod4, numel4);

  return 0;
}

TEST(SaveLoadCombineOp, XPU) {
  phi::XPUPlace xpu_place;
  phi::CPUPlace cpu_place;

  int r = SaveLoadCombineOpTest<phi::XPUPlace, float, float>(xpu_place);
  EXPECT_EQ(r, 0);
  r = SaveLoadCombineOpTest<phi::CPUPlace, float, float>(cpu_place);
  EXPECT_EQ(r, 0);

  r = SaveLoadCombineOpTest<phi::XPUPlace, int, int>(xpu_place);
  EXPECT_EQ(r, 0);
  r = SaveLoadCombineOpTest<phi::CPUPlace, int, int>(cpu_place);
  EXPECT_EQ(r, 0);

  r = SaveLoadCombineOpTest<phi::XPUPlace,
                            phi::dtype::float16,
                            phi::dtype::float16>(xpu_place);
  EXPECT_EQ(r, 0);
  r = SaveLoadCombineOpTest<phi::CPUPlace,
                            phi::dtype::float16,
                            phi::dtype::float16>(cpu_place);
  EXPECT_EQ(r, 0);

  r = SaveLoadCombineOpTest<phi::XPUPlace,
                            phi::dtype::bfloat16,
                            phi::dtype::bfloat16>(xpu_place);
  EXPECT_EQ(r, 0);
  r = SaveLoadCombineOpTest<phi::CPUPlace,
                            phi::dtype::bfloat16,
                            phi::dtype::bfloat16>(cpu_place);
  EXPECT_EQ(r, 0);
}
