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

#include "paddle/fluid/lite/kernels/arm/conv_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(conv_arm, retrive_op) {
  auto conv =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("conv");
  ASSERT_FALSE(conv.empty());
  ASSERT_TRUE(conv.front());
}

TEST(conv_arm, init) {
  ConvCompute conv;
  ASSERT_EQ(conv.precision(), PRECISION(kFloat));
  ASSERT_EQ(conv.target(), TARGET(kARM));
}

TEST(conv_arm, compare_test) {
  // TODO(xxx): add more compare
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv, kARM, kFloat, kNCHW, def);
