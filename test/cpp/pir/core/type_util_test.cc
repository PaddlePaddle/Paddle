// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/type_utils.h"

TEST(type_util_test, verify_compatible_dims) {
  EXPECT_TRUE(
      pir::VerifyCompatibleDims({pir::ShapedTypeInterface::kDynamic, 2, 2}));
  EXPECT_FALSE(pir::VerifyCompatibleDims({2, 3}));
}
