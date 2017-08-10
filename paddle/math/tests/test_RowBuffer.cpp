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
#include "paddle/math/RowBuffer.h"

TEST(RowBuffer, testAutoGrow) {
  paddle::RowBuffer buf(128);
  ASSERT_EQ(128UL, buf.getWidth());
  ASSERT_TRUE(buf.isAutoGrowth());
  buf.resize(2);
  ASSERT_EQ(2UL, buf.getRowCount());
  for (size_t i = 0; i < buf.getWidth() * 2; ++i) {
    buf.data()[i] = i;
  }
  for (size_t i = 0; i < buf.getRowCount(); ++i) {
    for (size_t j = 0; j < buf.getWidth(); ++j) {
      ASSERT_NEAR(i * buf.getWidth() + j, buf.get(i)[j], 1e-5);
    }
  }

  auto data = buf.getWithAutoGrowth(2);
  for (size_t i = 0; i < buf.getWidth(); ++i) {
    data[i] = i;
  }

  ASSERT_EQ(3UL, buf.getRowCount());
  for (size_t i = 0; i < buf.getRowCount() - 1; ++i) {
    for (size_t j = 0; j < buf.getWidth(); ++j) {
      ASSERT_NEAR(i * buf.getWidth() + j, buf.get(i)[j], 1e-5);
    }
  }
  for (size_t i = 0; i < buf.getWidth(); ++i) {
    ASSERT_NEAR(i, buf.get(2)[i], 1e-5);
  }
}

TEST(RowBuffer, testWithMemBuf) {
  paddle::CpuMemHandlePtr mem =
      std::make_shared<paddle::CpuMemoryHandle>(128 * 2 * sizeof(real));
  paddle::RowBuffer buf(mem, 128);
  ASSERT_TRUE(!buf.isAutoGrowth());
  ASSERT_EQ(2UL, buf.getRowCount());
  for (size_t i = 0; i < buf.getWidth() * 2; ++i) {
    buf.data()[i] = i;
  }
  for (size_t i = 0; i < buf.getRowCount(); ++i) {
    for (size_t j = 0; j < buf.getWidth(); ++j) {
      ASSERT_NEAR(i * buf.getWidth() + j, buf.getWithAutoGrowth(i)[j], 1e-5);
    }
  }

  ASSERT_DEATH_IF_SUPPORTED(buf.getWithAutoGrowth(3), ".*");
}
