/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

using bfloat16 = paddle::platform::bfloat16;

TEST(bfloat16, conversion_cpu) {
  // Conversion from float
  EXPECT_EQ(bfloat16(1.0f).x, 0x3f80);
  EXPECT_EQ(bfloat16(0.5f).x, 0x3f00);
  EXPECT_EQ(bfloat16(0.33333f).x, 0x3eaa);
  EXPECT_EQ(bfloat16(0.0f).x, 0x0000);
  EXPECT_EQ(bfloat16(-0.0f).x, 0x8000);
  EXPECT_EQ(bfloat16(65504.0f).x, 0x477f);
  EXPECT_EQ(bfloat16(65536.0f).x, 0x4780);

  // Conversion from double
  EXPECT_EQ(bfloat16(1.0).x, 0x3f80);
  EXPECT_EQ(bfloat16(0.5).x, 0x3f00);
  EXPECT_EQ(bfloat16(0.33333).x, 0x3eaa);
  EXPECT_EQ(bfloat16(0.0).x, 0x0000);
  EXPECT_EQ(bfloat16(-0.0).x, 0x8000);
  EXPECT_EQ(bfloat16(65504.0).x, 0x477f);
  EXPECT_EQ(bfloat16(65536.0).x, 0x4780);

  // Conversion from int
  EXPECT_EQ(bfloat16(-1).x, 0xbf80);
  EXPECT_EQ(bfloat16(0).x, 0x0000);
  EXPECT_EQ(bfloat16(1).x, 0x3f80);
  EXPECT_EQ(bfloat16(2).x, 0x4000);
  EXPECT_EQ(bfloat16(3).x, 0x4040);

  // Conversion from bool
  EXPECT_EQ(bfloat16(true).x, 0x3f80);
  EXPECT_EQ(bfloat16(false).x, 0x0000);

  // Assignment operator
  bfloat16 v_assign;
  v_assign = bfloat16(0.f);
  EXPECT_EQ(v_assign.x, 0x0000);
  v_assign = 0.5f;
  EXPECT_EQ(v_assign.x, 0x3f00);
  v_assign = 0.33333;
  EXPECT_EQ(v_assign.x, 0x3eaa);
  v_assign = -1;
  EXPECT_EQ(v_assign.x, 0xbf80);

  // Conversion operator
  EXPECT_EQ(static_cast<float>(bfloat16(0.5f)), 0.5f);
  EXPECT_NEAR(static_cast<double>(bfloat16(0.33333)), 0.33333, 0.01);
  EXPECT_EQ(static_cast<int>(bfloat16(-1)), -1);
  EXPECT_EQ(static_cast<bool>(bfloat16(true)), true);
}

TEST(bfloat16, arithmetic_cpu) {
  EXPECT_NEAR(static_cast<float>(bfloat16(1) + bfloat16(1)), 2, 0.001);
  EXPECT_EQ(static_cast<float>(bfloat16(5) + bfloat16(-5)), 0);
  EXPECT_NEAR(static_cast<float>(bfloat16(0.33333f) + bfloat16(0.66667f)), 1.0f,
              0.01);
  EXPECT_EQ(static_cast<float>(bfloat16(3) - bfloat16(5)), -2);
  EXPECT_NEAR(static_cast<float>(bfloat16(0.66667f) - bfloat16(0.33333f)),
              0.33334f, 0.01);
  EXPECT_NEAR(static_cast<float>(bfloat16(3.3f) * bfloat16(2.0f)), 6.6f, 0.01);
  EXPECT_NEAR(static_cast<float>(bfloat16(-2.1f) * bfloat16(-3.0f)), 6.3f, 0.1);
  EXPECT_NEAR(static_cast<float>(bfloat16(2.0f) / bfloat16(3.0f)), 0.66667f,
              0.01);
  EXPECT_EQ(static_cast<float>(bfloat16(1.0f) / bfloat16(2.0f)), 0.5f);
  EXPECT_EQ(static_cast<float>(-bfloat16(512.0f)), -512.0f);
  EXPECT_EQ(static_cast<float>(-bfloat16(-512.0f)), 512.0f);
}

TEST(bfloat16, comparison_cpu) {
  EXPECT_TRUE(bfloat16(1.0f) == bfloat16(1.0f));
  EXPECT_FALSE(bfloat16(-1.0f) == bfloat16(-0.5f));
  EXPECT_TRUE(bfloat16(1.0f) != bfloat16(0.5f));
  EXPECT_FALSE(bfloat16(-1.0f) != bfloat16(-1.0f));
  EXPECT_TRUE(bfloat16(1.0f) < bfloat16(2.0f));
  EXPECT_FALSE(bfloat16(-1.0f) < bfloat16(-1.0f));
  EXPECT_TRUE(bfloat16(1.0f) <= bfloat16(1.0f));
  EXPECT_TRUE(bfloat16(2.0f) > bfloat16(1.0f));
  EXPECT_FALSE(bfloat16(-2.0f) > bfloat16(-2.0f));
  EXPECT_TRUE(bfloat16(2.0f) >= bfloat16(2.0f));
}

TEST(bfloat16, lod_tensor_cpu) {
  framework::LoDTensor lod_tensor;

  std::vector<bfloat16> input_data = {bfloat16(1.0f), bfloat16(0.5f),
                                      bfloat16(0.33333f), bfloat16(0.0f)};
  EXPECT_EQ(input_data[0].x, 0x3f80);
  EXPECT_EQ(input_data[1].x, 0x3f00);
  EXPECT_EQ(input_data[2].x, 0x3eaa);
  EXPECT_EQ(input_data[3].x, 0x0000);

  lod_tensor.Resize({4, 1});
  lod_tensor.set_lod(framework::LoD({{0, 2, 4}}));
  bfloat16* data_ptr = lod_tensor.mutable_data<bfloat16>(CPUPlace());

  EXPECT_NE(data_ptr, nullptr);
  EXPECT_EQ(input_data.size(), static_cast<size_t>(lod_tensor.numel()));
  for (size_t i = 0; i < input_data.size(); ++i) {
    data_ptr[i] = input_data[i];
    EXPECT_EQ(data_ptr[i].x, input_data[i].x);
  }
}

TEST(bfloat16, floating) {
  // compile time assert.
  PADDLE_ENFORCE_EQ(
      std::is_floating_point<bfloat16>::value, true,
      platform::errors::Fatal("std::is_floating_point with bfloat16 data type "
                              "should be equal to true but it is not"));
}

TEST(bfloat16, print) {
  bfloat16 a = bfloat16(1.0f);
  std::cout << "a:" << a << std::endl;
  std::stringstream ss1, ss2;
  ss1 << a;
  ss2 << 1.0f;
  EXPECT_EQ(ss1.str(), ss2.str());
}

// CPU test
TEST(bfloat16, isinf) {
  bfloat16 a;
  a.x = 0x7f80;
  bfloat16 b = bfloat16(INFINITY);
  bfloat16 c = static_cast<bfloat16>(INFINITY);
  EXPECT_EQ(std::isinf(a), true);
  EXPECT_EQ(std::isinf(b), true);
  EXPECT_EQ(std::isinf(c), true);
}

TEST(bfloat16, isnan) {
  bfloat16 a;
  a.x = 0x7fff;
  bfloat16 b = bfloat16(NAN);
  bfloat16 c = static_cast<bfloat16>(NAN);
  EXPECT_EQ(std::isnan(a), true);
  EXPECT_EQ(std::isnan(b), true);
  EXPECT_EQ(std::isnan(c), true);
}

}  // namespace platform
}  // namespace paddle
