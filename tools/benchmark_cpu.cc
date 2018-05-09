//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "tools/benchmark.h"

using paddle::framework;
using paddle::platform;

void test_mul_op() {
  Scope scope;
  CPUPlace place;
  {
    auto var = scope->Var("X");
    auto x = var->GetMutable<LoDTensor>();
    x->Resize({10, 10});
    float *expect = x->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }
  {
    auto var = scope->Var("Y");
    auto x = var->GetMutable<LoDTensor>();
    x->Resize({10, 10});
    float *expect = x->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }

  auto out_var = scope->Var("Out");
  auto out_tensor = out_var->GetMutable<f::LoDTensor>();
  AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(
      "dropout", {{"X", {"X"}}, {"Y", {"Y"}}}, {{"Out", {"Out"}}}, attrs);
  op->Run(scope, place);

  // check output
}

int main() {
  test_mul_op();
  return 0;
}
