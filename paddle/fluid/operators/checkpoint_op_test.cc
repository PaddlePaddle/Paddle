/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

USE_NO_KERNEL_OP(checkpoint_save)

TEST(CheckpointSaveOp, CPU) {
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
  attrs.insert({"dir", std::string("tensor/ckpt")});

  auto save_op = paddle::framework::OpRegistry::CreateOp(
      "checkpoint_save", {{"X", {"test_var"}}}, {}, attrs);
  save_op->Run(scope, place);
}
