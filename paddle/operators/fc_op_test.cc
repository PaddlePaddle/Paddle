/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
namespace f = paddle::framework;

USE_OP_WITHOUT_KERNEL(fc);

TEST(FC, create) {
  for (size_t i = 0; i < 1000000; ++i) {
    auto tmp = f::OpRegistry::CreateOp("fc", {"X", "W", "B"}, {"O"}, {});
    ASSERT_NE(tmp, nullptr);
  }
}