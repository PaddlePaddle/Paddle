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

#include "paddle/fluid/lite/operators/softmax_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(softmax_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({10, 20})));
  output->Resize(DDim(std::vector<int64_t>{10, 20}));

  // set data
  for (int i = 0; i < 10 * 20; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 10 * 20; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("softmax");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"output"});
  desc.SetAttr("axis", static_cast<int>(-1));

  SoftmaxOp softmax("softmax");

  softmax.SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)}});
  softmax.Attach(desc, &scope);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
