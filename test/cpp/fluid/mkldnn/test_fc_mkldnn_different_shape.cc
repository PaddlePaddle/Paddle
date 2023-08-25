/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(fc);

paddle::framework::VarDesc *Data(
    paddle::framework::BlockDesc *block,
    std::string name,
    std::vector<int64_t> shape = {},
    bool is_persistable = false,
    paddle::framework::proto::VarType::Type data_type =
        paddle::framework::proto::VarType::FP32) {
  auto *var = block->Var(name);
  var->SetType(paddle::framework::proto::VarType::LOD_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}
TEST(FCMklDNNOp, ChangeSrcShape) {
  paddle::platform::Place place = paddle::platform::CPUPlace();
  paddle::framework::Scope scope;
  paddle::framework::ProgramDesc program;

  auto *block = program.MutableBlock(0);
  auto *x = Data(block, "Input", {2, 3});
  auto *w = Data(block, "W", {3, 2});
  auto *out = Data(block, "Out", {2, 2});
  auto *fc_op = block->AppendOp();
  auto x_value = scope.Var(x->Name())->GetMutable<phi::DenseTensor>();
  x_value->Resize({2, 3});

  auto *w_value = scope.Var(w->Name())->GetMutable<phi::DenseTensor>();
  w_value->Resize({3, 2});
  float a_original_arr[] = {0, 1, 2, 3, 4, 5};
  float b_original_arr[] = {0.0, .1, .2, .3, .4, .5};

  std::copy_n(a_original_arr, 4, x_value->mutable_data<float>(place));
  std::copy_n(b_original_arr, 4, w_value->mutable_data<float>(place));

  auto *var_out = scope.Var(out->Name())->GetMutable<phi::DenseTensor>();
  var_out->Resize({2, 2});
  auto *out_data = var_out->mutable_data<float>(paddle::platform::CPUPlace());
  for (int i = 0; i < 4; ++i) {
    out_data[i] = i;
  }

  fc_op->SetType("fc");
  fc_op->SetInput("Input", {x->Name()});
  fc_op->SetInput("W", {w->Name()});
  fc_op->SetOutput("Out", {out->Name()});
  fc_op->SetAttr("in_num_col_dims", 1);
  fc_op->SetAttr("use_mkldnn", {true});
  paddle::framework::NaiveExecutor exe(place);
  exe.CreateVariables(program, 0, true, &scope);
  exe.Prepare(&scope, program, 0, false);
  exe.Run();

  auto *x_tensor = exe.FindTensor("Input");
  auto *w_tensor = exe.FindTensor("W");

  x_tensor->Resize({2, 2});
  w_tensor->Resize({2, 2});

  float a_arr[] = {0, 1, 2, 3};
  float b_arr[] = {0.0, .1, .2, .3};
  float c_arr[] = {.2, .3, .6, 1.1};

  std::copy_n(a_arr, 4, x_tensor->mutable_data<float>(place));
  std::copy_n(b_arr, 4, w_tensor->mutable_data<float>(place));
  exe.Run();
  auto *out_tensor = exe.FindTensor("Out");
  auto *c_data = out_tensor->mutable_data<float>(place);
  for (int i = 0; i < 4; i++) {
    CHECK_EQ(c_data[i], c_arr[i]) << "Fc output is wrong value!";
  }
}
TEST(FCMklDNNOp, ChangeSrcLayout) {
  paddle::platform::Place place = paddle::platform::CPUPlace();
  paddle::framework::Scope scope;
  paddle::framework::ProgramDesc program;

  auto *block = program.MutableBlock(0);
  auto *x = Data(block, "Input", {2, 1, 2});
  auto *w = Data(block, "W", {2, 2});
  auto *out = Data(block, "Out", {2, 2});
  auto *fc_op = block->AppendOp();

  auto x_value = scope.Var(x->Name())->GetMutable<phi::DenseTensor>();
  auto *w_value = scope.Var(w->Name())->GetMutable<phi::DenseTensor>();
  x_value->Resize({2, 1, 2});
  w_value->Resize({2, 2});

  float a_original_arr[] = {0, 1, 0, 1};
  float b_original_arr[] = {0.0, .1, 0.0, .1};

  std::copy_n(a_original_arr, 4, x_value->mutable_data<float>(place));
  std::copy_n(b_original_arr, 4, w_value->mutable_data<float>(place));

  auto *var_out = scope.Var(out->Name())->GetMutable<phi::DenseTensor>();
  var_out->Resize({2, 2});
  auto *out_data = var_out->mutable_data<float>(paddle::platform::CPUPlace());
  for (int i = 0; i < 4; ++i) {
    out_data[i] = i;
  }

  fc_op->SetType("fc");
  fc_op->SetInput("Input", {x->Name()});
  fc_op->SetInput("W", {w->Name()});
  fc_op->SetOutput("Out", {out->Name()});
  fc_op->SetAttr("in_num_col_dims", 1);
  fc_op->SetAttr("use_mkldnn", {true});
  paddle::framework::NaiveExecutor exe(place);
  exe.CreateVariables(program, 0, true, &scope);
  exe.Prepare(&scope, program, 0, false);
  exe.Run();

  auto *x_tensor = exe.FindTensor("Input");
  auto *w_tensor = exe.FindTensor("W");

  x_tensor->set_layout(phi::DataLayout::NHWC);
  w_tensor->set_layout(phi::DataLayout::NHWC);

  exe.Run();
}
