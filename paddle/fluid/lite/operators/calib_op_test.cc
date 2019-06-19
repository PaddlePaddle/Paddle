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

#include "paddle/fluid/lite/operators/calib_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

#ifdef LITE_WITH_ARM
TEST(calib_op_lite, TestARM) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("Input")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({1, 10, 20})));
  output->Resize(DDim(std::vector<int64_t>{1, 10, 20}));

  // set data
  for (int i = 0; i < 10 * 20; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 10 * 20; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("calib");
  desc.SetInput("Input", {"Input"});
  desc.SetOutput("Out", {"output"});
  desc.SetAttr("in_dtype", static_cast<int>(PRECISION(kInt8)));
  desc.SetAttr("out_dtype", static_cast<int>(PRECISION(kFloat)));
  desc.SetAttr("in_scale", 10.0f);

  CalibOpLite calib("calib");

  calib.SetValidPlaces({Place{TARGET(kARM), PRECISION(kAny)}});
  calib.Attach(desc, &scope);
  auto kernels = calib.CreateKernels({Place{TARGET(kARM), PRECISION(kAny)}});
  ASSERT_FALSE(kernels.empty());
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

#ifdef LITE_WITH_ARM
USE_LITE_KERNEL(calib, kARM, kAny, kAny, def);
#endif
