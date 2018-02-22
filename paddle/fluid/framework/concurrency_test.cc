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

#include <thread>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/block_desc.h"

USE_NO_KERNEL_OP(go);
USE_NO_KERNEL_OP(sum);

namespace f = paddle::framework;
namespace p = paddle::platform;

void InitTensorsInScope(f::Scope &scope, p::CPUPlace &place) {
  p::CPUDeviceContext ctx(place);
  for (int i = 0; i < 2; ++i) {
    auto var_name = paddle::string::Sprintf("x%d", i);
    auto var = scope.Var(var_name);
    auto tensor = var->GetMutable<f::LoDTensor>();
    tensor->Resize({10, 10});
    float *expect = tensor->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<f::LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate
}

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           f::BlockDesc *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      block->Var(v);
      // var->SetDataType(f::proto::DataType::FP32);
    }
  }

  // insert op
  auto op = block->AppendOp();
  op->SetType(type);
  for (auto &kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto &kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
}

TEST(Concurrency, Go_Op) {
  f::Scope scope;
  p::CPUPlace place;

  InitTensorsInScope(scope, place);

  f::ProgramDesc program;
  f::BlockDesc *block = program.MutableBlock(0);

  AddOp("sum", {{"X", {"x0", "x1"}}}, {{"Out", {"Out"}}}, {}, block);

  f::AttributeMap attrs;
  attrs.insert({"sub_block", block});

  auto go_op = f::OpRegistry::CreateOp("go", {{"X", {"x0", "x1"}}},
                                       {{"Out", {"Out"}}}, attrs);
  std::cout << "GOING INTO THREAD" << std::endl << std::flush;
  go_op->Run(scope, place);
  std::cout << "GOING INTO SLEEP" << std::endl << std::flush;
  usleep(100000000);
  std::cout << "GOING DONE" << std::endl << std::flush;
}
