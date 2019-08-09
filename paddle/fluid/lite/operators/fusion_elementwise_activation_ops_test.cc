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

#include "paddle/fluid/lite/operators/fusion_elementwise_activation_ops.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(fusion_elementwise_activation_op_lite, test) {
  // prepare variables
  lite::Scope scope;
  auto* x = scope.Var("x")->GetMutable<lite::Tensor>();
  auto* y = scope.Var("y")->GetMutable<lite::Tensor>();
  auto* out = scope.Var("out")->GetMutable<lite::Tensor>();
  x->Resize(lite::DDim(std::vector<int64_t>({10, 20})));
  y->Resize(lite::DDim(std::vector<int64_t>({10, 20})));
  out->Resize(lite::DDim(std::vector<int64_t>{10, 20}));

  // set data
  for (int i = 0; i < 10 * 20; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 10 * 20; i++) {
    y->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 10 * 20; i++) {
    out->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("fusion_elementwise_add_activation");
  desc.SetInput("X", {"x"});
  desc.SetInput("Y", {"y"});
  desc.SetOutput("Out", {"out"});
  desc.SetAttr("axis", static_cast<int>(1));
  desc.SetAttr("act_type", std::string("relu"));

  FusionElementwiseActivationOp fuse_op("fusion_elementwise_add_activation");

  fuse_op.SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)}});
  fuse_op.Attach(desc, &scope);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
