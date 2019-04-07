// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/operators/fc_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(fc_op_lite, test) {
  LOG(INFO) << "\n" << KernelRegistry::Global().DebugString();
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* w = scope.Var("w")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  x->Resize({1, 10, 20});
  w->Resize({20, 20});
  bias->Resize({1, 10});
  output->Resize({10, 20});

  // set data
  for (int i = 0; i < 10 * 20; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 20 * 20; i++) {
    w->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 1 * 10; i++) {
    bias->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 10 * 20; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  framework::OpDesc desc;
  desc.SetType("fc");
  desc.SetInput("Input", {"x"});
  desc.SetInput("W", {"w"});
  desc.SetInput("Bias", {"bias"});
  desc.SetOutput("Out", {"output"});
  desc.SetAttr("in_num_col_dims", static_cast<int>(1));

  FcOpLite fc("fc");

  fc.SetValidPlaces({OpLite::Place{TARGET(kHost), PRECISION(kFloat)}});
  fc.PickKernel({OpLite::Place{TARGET(kHost), PRECISION(kFloat)}});

  fc.Attach(desc, &scope);
  fc.Run();

  for (int i = 0; i < 10 * 20; i++) {
    LOG(INFO) << output->data<float>()[i];
  }
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kHost, kFloat);
