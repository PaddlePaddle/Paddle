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

#include "paddle/fluid/lite/operators/transpose_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

// Transpose
TEST(transpose_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  const int h = 10;
  const int w = 20;
  x->Resize(DDim(std::vector<int64_t>({h, w})));
  output->Resize(DDim(std::vector<int64_t>{w, h}));

  // set data
  for (int i = 0; i < h * w; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < w * h; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"output"});
  // axis change for shape in mobilenetssd: [1, 24, 2, 2] => [1, 2, 2, 24]
  std::vector<int> axis{0, 2, 3, 1};
  desc.SetAttr("axis", axis);

  TransposeOp transpose("transpose");

  transpose.SetValidPlaces({Place{TARGET(kARM), PRECISION(kFloat)}});
  transpose.Attach(desc, &scope);
}

// Transpose2
TEST(transpose2_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  const int h = 10;
  const int w = 20;
  x->Resize(DDim(std::vector<int64_t>({h, w})));
  output->Resize(DDim(std::vector<int64_t>{w, h}));

  // set data
  for (int i = 0; i < h * w; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < w * h; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("transpose2");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"output"});
  // axis change for shape in mobilenetssd: [1, 24, 2, 2] => [1, 2, 2, 24]
  std::vector<int> axis{0, 2, 3, 1};
  desc.SetAttr("axis", axis);

  Transpose2Op transpose2("transpose2");

  transpose2.SetValidPlaces({Place{TARGET(kARM), PRECISION(kFloat)}});
  transpose2.Attach(desc, &scope);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
