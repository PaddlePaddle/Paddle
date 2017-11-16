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
#include "paddle/math/float16.h"

namespace paddle {

#ifdef PADDLE_CUDA_FP16
TEST(float16, conversion_gpu) {
  // Conversion to and from cuda half
  float16 v1 = half(float16(1.0f));
  EXPECT_EQ(v1.x, 0x3c00);

  // Conversion to and from Eigen::half
  float16 v2 = Eigen::half(float16(0.5f));
  EXPECT_EQ(v2.x, 0x3800);

  // Conversion from float
  EXPECT_EQ(float16(1.0f).x, 0x3c00);
  EXPECT_EQ(float16(0.5f).x, 0x3800);
  EXPECT_EQ(float16(0.33333f).x, 0x3555);
  EXPECT_EQ(float16(0.0f).x, 0x0000);
  EXPECT_EQ(float16(-0.0f).x, 0x8000);
  EXPECT_EQ(float16(65504.0f).x, 0x7bff);
  EXPECT_EQ(float16(65536.0f).x, 0x7c00);

  // Conversion from double

  // Conversion from int

  // Conversion from bool
}

TEST(float16, arithmetic_gpu) { EXPECT_EQ(float(float16(2) + float16(2)), 4); }

TEST(float16, comparison_gpu) { EXPECT_TRUE(float16(1.0f) > float16(0.5f)); }
#endif

TEST(float16, conversion_cpu) {
  // Conversion to and from Eigen::half
  EXPECT_EQ(float16(Eigen::half(float16(1.0f))).x, 0x3c00);
  EXPECT_EQ(float16(Eigen::half(float16(0.5f))).x, 0x3800);
  EXPECT_EQ(float16(Eigen::half(float16(0.33333f))).x, 0x3555);
  EXPECT_EQ(float16(Eigen::half(float16(0.0f))).x, 0x0000);
  EXPECT_EQ(float16(Eigen::half(float16(-0.0f))).x, 0x8000);
  EXPECT_EQ(float16(Eigen::half(float16(65504.0f))).x, 0x7bff);
  EXPECT_EQ(float16(Eigen::half(float16(65536.0f))).x, 0x7c00);

  // Conversion from float
  EXPECT_EQ(float16(1.0f).x, 0x3c00);
  EXPECT_EQ(float16(0.5f).x, 0x3800);
  EXPECT_EQ(float16(0.33333f).x, 0x3555);
  EXPECT_EQ(float16(0.0f).x, 0x0000);
  EXPECT_EQ(float16(-0.0f).x, 0x8000);
  EXPECT_EQ(float16(65504.0f).x, 0x7bff);
  EXPECT_EQ(float16(65536.0f).x, 0x7c00);

  // Conversion from double

  // Conversion from int

  // Conversion from bool
}

TEST(float16, arithmetic_cpu) { EXPECT_EQ(float(float16(2) + float16(2)), 4); }

TEST(float16, comparison_cpu) { EXPECT_TRUE(float16(1.0f) > float16(0.5f)); }

}  // namespace paddle
