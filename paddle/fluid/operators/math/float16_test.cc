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

#include "paddle/math/float16.h"

#include <gtest/gtest.h>

namespace paddle {

TEST(float16, conversion_cpu) {
  // Explicit conversion from Eigen::half
  EXPECT_EQ(float16(Eigen::half(1.0f)).x, 0x3c00);
  EXPECT_EQ(float16(Eigen::half(0.5f)).x, 0x3800);
  EXPECT_EQ(float16(Eigen::half(0.33333f)).x, 0x3555);
  EXPECT_EQ(float16(Eigen::half(0.0f)).x, 0x0000);
  EXPECT_EQ(float16(Eigen::half(-0.0f)).x, 0x8000);
  EXPECT_EQ(float16(Eigen::half(65504.0f)).x, 0x7bff);
  EXPECT_EQ(float16(Eigen::half(65536.0f)).x, 0x7c00);

  // Conversion from float
  EXPECT_EQ(float16(1.0f).x, 0x3c00);
  EXPECT_EQ(float16(0.5f).x, 0x3800);
  EXPECT_EQ(float16(0.33333f).x, 0x3555);
  EXPECT_EQ(float16(0.0f).x, 0x0000);
  EXPECT_EQ(float16(-0.0f).x, 0x8000);
  EXPECT_EQ(float16(65504.0f).x, 0x7bff);
  EXPECT_EQ(float16(65536.0f).x, 0x7c00);

  // Conversion from double
  EXPECT_EQ(float16(1.0).x, 0x3c00);
  EXPECT_EQ(float16(0.5).x, 0x3800);
  EXPECT_EQ(float16(0.33333).x, 0x3555);
  EXPECT_EQ(float16(0.0).x, 0x0000);
  EXPECT_EQ(float16(-0.0).x, 0x8000);
  EXPECT_EQ(float16(65504.0).x, 0x7bff);
  EXPECT_EQ(float16(65536.0).x, 0x7c00);

  // Conversion from int
  EXPECT_EQ(float16(-1).x, 0xbc00);
  EXPECT_EQ(float16(0).x, 0x0000);
  EXPECT_EQ(float16(1).x, 0x3c00);
  EXPECT_EQ(float16(2).x, 0x4000);
  EXPECT_EQ(float16(3).x, 0x4200);

  // Conversion from bool
  EXPECT_EQ(float16(true).x, 0x3c00);
  EXPECT_EQ(float16(false).x, 0x0000);

  // Default constructor
  float16 v_def;
  EXPECT_EQ(v_def.x, 0x0000);

  // Assignment operator
  float16 v_assign;
  v_assign = v_def;
  EXPECT_EQ(v_assign.x, 0x0000);
  v_assign = Eigen::half(1.0f);
  EXPECT_EQ(v_assign.x, 0x3c00);
  v_assign = 0.5f;
  EXPECT_EQ(v_assign.x, 0x3800);
  v_assign = 0.33333;
  EXPECT_EQ(v_assign.x, 0x3555);
  v_assign = -1;
  EXPECT_EQ(v_assign.x, 0xbc00);
  v_assign = true;
  EXPECT_EQ(v_assign.x, 0x3c00);

  // Conversion operator
  EXPECT_EQ(Eigen::half(float16(1.0f)).x, 0x3c00);
  EXPECT_EQ(float(float16(0.5f)), 0.5f);
  EXPECT_NEAR(double(float16(0.33333)), 0.33333, 0.0001);
  EXPECT_EQ(int(float16(-1)), -1);
  EXPECT_EQ(bool(float16(true)), true);
}

TEST(float16, arithmetic_cpu) {
  EXPECT_EQ(float(float16(1) + float16(1)), 2);
  EXPECT_EQ(float(float16(5) + float16(-5)), 0);
  EXPECT_NEAR(float(float16(0.33333f) + float16(0.66667f)), 1.0f, 0.001);
  EXPECT_EQ(float(float16(3) - float16(5)), -2);
  EXPECT_NEAR(float(float16(0.66667f) - float16(0.33333f)), 0.33334f, 0.001);
  EXPECT_NEAR(float(float16(3.3f) * float16(2.0f)), 6.6f, 0.01);
  EXPECT_NEAR(float(float16(-2.1f) * float16(-3.0f)), 6.3f, 0.01);
  EXPECT_NEAR(float(float16(2.0f) / float16(3.0f)), 0.66667f, 0.001);
  EXPECT_EQ(float(float16(1.0f) / float16(2.0f)), 0.5f);
  EXPECT_EQ(float(-float16(512.0f)), -512.0f);
  EXPECT_EQ(float(-float16(-512.0f)), 512.0f);
}

TEST(float16, comparison_cpu) {
  EXPECT_TRUE(float16(1.0f) == float16(1.0f));
  EXPECT_FALSE(float16(-1.0f) == float16(-0.5f));
  EXPECT_TRUE(float16(1.0f) != float16(0.5f));
  EXPECT_FALSE(float16(-1.0f) != float16(-1.0f));
  EXPECT_TRUE(float16(1.0f) < float16(2.0f));
  EXPECT_FALSE(float16(-1.0f) < float16(-1.0f));
  EXPECT_TRUE(float16(1.0f) <= float16(1.0f));
  EXPECT_TRUE(float16(2.0f) > float16(1.0f));
  EXPECT_FALSE(float16(-2.0f) > float16(-2.0f));
  EXPECT_TRUE(float16(2.0f) >= float16(2.0f));

  EXPECT_TRUE(float16(0.0f) == float16(-0.0f));
  EXPECT_TRUE(float16(0.0f) <= float16(-0.0f));
  EXPECT_TRUE(float16(0.0f) >= float16(-0.0f));
  EXPECT_FALSE(float16(0.0f) < float16(-0.0f));
  EXPECT_FALSE(float16(-0.0f) < float16(0.0f));
  EXPECT_FALSE(float16(0.0f) > float16(-0.0f));
  EXPECT_FALSE(float16(-0.0f) > float16(0.0f));
}

}  // namespace paddle
