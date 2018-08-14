// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/random/philox_engine.h"
#include "gtest/gtest.h"

TEST(philox, large_int) {
  using Int128 = paddle::platform::random::LargeInt<4>;

  Int128 a;
  a = 10u;
  a += UINT32_MAX;
  uint64_t tmp;
  tmp = a[1];
  tmp <<= 32;
  tmp |= a[0];

  ASSERT_EQ(tmp, 10UL + UINT32_MAX);
  ++a;
  tmp = a[1];
  tmp <<= 32;
  tmp |= a[0];
  ASSERT_EQ(tmp, 11UL + UINT32_MAX);
}

TEST(philox, discard) {
  using Engine = paddle::platform::random::Philox32x4;
  uint64_t seed = 10, discard = 193845;
  Engine engine1(seed);
  for (uint64_t i = 0; i < discard; ++i) {
    engine1();
  }
  Engine engine2(seed);
  engine2.Discard(discard);

  for (size_t i = 0; i < 10; ++i) {
    auto val1 = engine1();
    auto val2 = engine2();

    for (size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(val1[j], val2[j]);
    }
  }
}
