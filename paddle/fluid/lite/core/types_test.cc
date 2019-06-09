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

#include "paddle/fluid/lite/core/types.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {
namespace core {

TEST(KernelPickFactor, Default) {
  KernelPickFactor factor;
  ASSERT_FALSE(factor.IsTargetConsidered());
  ASSERT_FALSE(factor.IsPrecisionConsidered());
  ASSERT_FALSE(factor.IsDataLayoutConsidered());
}

TEST(KernelPickFactor, Set) {
  KernelPickFactor factor;
  factor.ConsiderTarget();
  ASSERT_TRUE(factor.IsTargetConsidered());
  factor.ConsiderPrecision();
  ASSERT_TRUE(factor.IsPrecisionConsidered());
  factor.ConsiderDataLayout();
  ASSERT_TRUE(factor.IsDataLayoutConsidered());

  LOG(INFO) << "factor " << factor;
}

}  // namespace core
}  // namespace lite
}  // namespace paddle
