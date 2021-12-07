//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/platform/place.h"

#include "gtest/gtest.h"

TEST(Place, Equality) {
  paddle::platform::CPUPlace cpu;
  paddle::platform::CUDAPlace g0(0), g1(1), gg0(0);
  paddle::platform::XPUPlace x0(0), x1(1), xx0(0);
  paddle::platform::MLUPlace m0(0), m1(1), mm0(0);

  EXPECT_EQ(cpu, cpu);
  EXPECT_EQ(g0, g0);
  EXPECT_EQ(g1, g1);
  EXPECT_EQ(g0, gg0);
  EXPECT_EQ(x0, x0);
  EXPECT_EQ(x1, x1);
  EXPECT_EQ(x0, xx0);
  EXPECT_EQ(m0, m0);
  EXPECT_EQ(m1, m1);
  EXPECT_EQ(m0, mm0);

  EXPECT_NE(g0, g1);
  EXPECT_NE(x0, x1);
  EXPECT_NE(m0, m1);

  EXPECT_TRUE(paddle::platform::places_are_same_class(g0, gg0));
  EXPECT_TRUE(paddle::platform::places_are_same_class(x0, xx0));
  EXPECT_FALSE(paddle::platform::places_are_same_class(g0, cpu));
  EXPECT_FALSE(paddle::platform::places_are_same_class(x0, cpu));
  EXPECT_FALSE(paddle::platform::places_are_same_class(g0, x0));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << paddle::platform::XPUPlace(1);
    EXPECT_EQ("XPUPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::platform::MLUPlace(1);
    EXPECT_EQ("MLUPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::platform::CUDAPlace(1);
    EXPECT_EQ("CUDAPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::platform::CPUPlace();
    EXPECT_EQ("CPUPlace", ss.str());
  }
}
