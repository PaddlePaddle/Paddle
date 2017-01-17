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

#include "TensorShape.h"
#include <gtest/gtest.h>

namespace paddle {

TEST(TensorShape, Constructor) {
  TensorShape t1;
  EXPECT_EQ(t1.ndims(), 0);
  EXPECT_EQ(t1.getElements(), 0);

  TensorShape t2(3);
  EXPECT_EQ(t2.ndims(), 3);
  EXPECT_EQ(t2.getElements(), 1);

  TensorShape t3({8, 10});
  EXPECT_EQ(t3.ndims(), 2);
  EXPECT_EQ(t3.getElements(), 80);

  TensorShape t4(t3);
  EXPECT_EQ(t4.ndims(), t3.ndims());
  EXPECT_EQ(t4.getElements(), t3.getElements());

  TensorShape t5({1, 2, 3, 4, 5});
  EXPECT_EQ(t5.ndims(), 5);
  EXPECT_EQ(t5.getElements(), 120);
}

TEST(TensorShape, GetAndSet) {
  TensorShape t({1, 2, 3});
  EXPECT_EQ(t.ndims(), 3);
  EXPECT_EQ(t.getElements(), 6);

  EXPECT_EQ(t[1], 2);
  t.setDim(1, 100);
  EXPECT_EQ(t.getElements(), 300);
  EXPECT_EQ(t[1], 100);
}

}  // namespace paddle
