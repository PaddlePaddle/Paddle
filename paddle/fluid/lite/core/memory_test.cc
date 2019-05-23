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

#include "paddle/fluid/lite/core/memory.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

TEST(memory, test) {
  auto* buf = TargetMalloc(TARGET(kX86), 10);
  ASSERT_TRUE(buf);
  TargetFree(TARGET(kX86), buf);

#ifdef LITE_WITH_CUDA
  auto* buf_cuda = TargetMalloc(TARGET(kCUDA), 10);
  ASSERT_TRUE(buf_cuda);
  TargetFree(Target(kCUDA), buf_cuda);
#endif
}

}  // namespace lite
}  // namespace paddle
