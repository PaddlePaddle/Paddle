// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA
int OriginalRoundToPowerOfTwo(int dim) {
  if (dim > 512) {
    return 1024;
  } else if (dim > 256) {
    return 512;
  } else if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}
#else
int OriginalRoundToPowerOfTwo(int dim) {
  // HIP results in error or nan if > 256
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}
#endif

TEST(threads_round_up, test_gpu) {
  std::vector<int> in_arr{17, 65, 127, 200, 312, 612};
  std::vector<int> out_arr(6, 0);
  for (size_t i = 0; i < in_arr.size(); ++i) {
    out_arr[i] = platform::RoundToPowerOfTwo(in_arr[i]);
    int result = OriginalRoundToPowerOfTwo(in_arr[i]);
    EXPECT_EQ(out_arr[i], result);
  }
}

}  // namespace platform
}  // namespace paddle
