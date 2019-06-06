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

#include "paddle/fluid/lite/operators/scale_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(scale_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({10, 20})));
  output->Resize(DDim(std::vector<int64_t>{1, 1}));

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("scale");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"output"});
  desc.SetAttr("bias_after_scale", false);
  desc.SetAttr("scale", 0.5f);
  desc.SetAttr("bias", 0.125f);

  ScaleOp scale("scale");

  scale.SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)}});
  scale.Attach(desc, &scope);
  scale.CheckShape();
  scale.InferShape();

  // check output dims
  auto x_dims = x->dims();
  auto output_dims = output->dims();
  CHECK_EQ(output_dims.size(), x_dims.size());
  for (size_t i = 0; i < output_dims.size(); i++) {
    CHECK_EQ(output_dims[i], x_dims[i]);
  }
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
