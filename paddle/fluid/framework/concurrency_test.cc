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
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

USE_NO_KERNEL_OP(go);
USE_NO_KERNEL_OP(sum);

namespace f = paddle::framework;
namespace p = paddle::platform;

namespace paddle {
namespace framework {
void InitTensorsInScope(Scope &scope, p::CPUPlace &place) {
  p::CPUDeviceContext ctx(place);
  for (int i = 0; i < 2; ++i) {
    auto var_name = paddle::string::Sprintf("x%d", i);
    auto var = scope.Var(var_name);
    auto tensor = var->GetMutable<LoDTensor>();
    tensor->Resize({10, 10});
    float *expect = tensor->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate
}

void AddOp(const std::string &type, const VariableNameMap &inputs,
           const VariableNameMap &outputs, AttributeMap attrs,
           BlockDesc *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(proto::VarType::FP32);
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
  Scope scope;
  p::CPUPlace place;

  InitTensorsInScope(scope, place);

  ProgramDesc program;
  BlockDesc *block = program.MutableBlock(0);

  AddOp("sum", {{"X", {"x0", "x1"}}}, {{"Out", {"Out"}}}, {}, block);

  AttributeMap attrs;
  attrs.insert({"sub_block", block});

  auto go_op = OpRegistry::CreateOp("go", {{"X", {"x0", "x1"}}},
                                    {{"Out", {"Out"}}}, attrs);
  go_op->Run(scope, place);
  usleep(1000000);  // TODO(thuan): Replace this command with channel receive
}
}  // namespace framework
}  // namespace paddle
