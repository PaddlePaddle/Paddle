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
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

template <typename Place, typename T>
int SaveLoadOpTest(Place place, int dim_1, int dim_2) {
  // use cpu place for ground truth
  phi::CPUPlace cpu_place;
  std::vector<T> ground_truth_cpu(dim_1 * dim_2);
  for (int i = 0; i < dim_1 * dim_2; i++) {
    ground_truth_cpu[i] = static_cast<T>(i);
  }

  // scope, var, tensor and lod
  paddle::framework::Scope scope;
  auto var = scope.Var("test_var");
  auto tensor = var->GetMutable<phi::DenseTensor>();
  tensor->Resize({dim_1, dim_2});
  phi::LegacyLoD expect_lod;
  expect_lod.resize(1);
  for (int i = 0; i < dim_1; i++) {
    expect_lod[0].push_back(i);
  }
  tensor->set_lod(expect_lod);
  T* src_mutable = tensor->mutable_data<T>(place);
  // copy cpu data to tensor
  paddle::memory::Copy(place,
                       src_mutable,
                       cpu_place,
                       ground_truth_cpu.data(),
                       sizeof(T) * ground_truth_cpu.size());

  // run save op
  paddle::framework::AttributeMap attrs;
  attrs.insert({"file_path", std::string("tensor.save")});
  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);

  // result var and tensor
  auto load_var = scope.Var("out_var");
  auto target = load_var->GetMutable<phi::DenseTensor>();

  // run load op
  auto load_op = paddle::framework::OpRegistry::CreateOp(
      "load", {}, {{"Out", {"out_var"}}}, attrs);
  load_op->Run(scope, place);

  // copy result tensor data to cpu
  T* actual = target->data<T>();
  std::vector<T> actual_cpu(dim_1 * dim_2);
  paddle::memory::Copy(cpu_place,
                       actual_cpu.data(),
                       place,
                       actual,
                       sizeof(T) * ground_truth_cpu.size());

  // check result: data
  for (int i = 0; i < dim_1 * dim_2; i++) {
    if (actual_cpu[i] != ground_truth_cpu[i]) {
      return 1;
    }
  }

  // check result: lod
  auto& actual_lod = target->lod();
  if (expect_lod.size() != actual_lod.size()) {
    return 1;
  }
  for (size_t i = 0; i < expect_lod.size(); ++i) {  // NOLINT
    for (size_t j = 0; j < expect_lod[i].size(); ++j) {
      if (expect_lod[i][j] != actual_lod[i][j]) {
        return 1;
      }
    }
  }
  return 0;
}

TEST(SaveLoadOp, XPU) {
  phi::XPUPlace xpu_place(0);
  phi::CPUPlace cpu_place;
  int r = 0;

  r = SaveLoadOpTest<phi::XPUPlace, float>(xpu_place, 3, 10);
  EXPECT_EQ(r, 0);
  r = SaveLoadOpTest<phi::CPUPlace, float>(cpu_place, 3, 10);
  EXPECT_EQ(r, 0);

  r = SaveLoadOpTest<phi::XPUPlace, int>(xpu_place, 2, 128);
  EXPECT_EQ(r, 0);
  r = SaveLoadOpTest<phi::CPUPlace, int>(cpu_place, 2, 128);
  EXPECT_EQ(r, 0);

  r = SaveLoadOpTest<phi::XPUPlace, phi::dtype::float16>(xpu_place, 2, 128);
  EXPECT_EQ(r, 0);
  r = SaveLoadOpTest<phi::CPUPlace, phi::dtype::float16>(cpu_place, 2, 128);
  EXPECT_EQ(r, 0);

  r = SaveLoadOpTest<phi::XPUPlace, phi::dtype::bfloat16>(xpu_place, 4, 32);
  EXPECT_EQ(r, 0);
  r = SaveLoadOpTest<phi::CPUPlace, phi::dtype::bfloat16>(cpu_place, 4, 32);
  EXPECT_EQ(r, 0);
}
