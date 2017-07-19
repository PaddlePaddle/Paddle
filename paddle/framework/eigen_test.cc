/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/framework/eigen.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {

TEST(EigenDim, From) {
  EigenDim<3>::Type ed = EigenDim<3>::From(make_ddim({1, 2, 3}));
  EXPECT_EQ(1, ed[0]);
  EXPECT_EQ(2, ed[1]);
  EXPECT_EQ(3, ed[2]);
}

TEST(Eigen, Tensor) {
  Tensor t;
  float* p = t.mutable_data<float>(make_ddim({1, 2, 3}), platform::CPUPlace());
  for (int i = 0; i < 1 * 2 * 3; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenTensor<float, 3>::Type et = EigenTensor<float, 3>::From(t);

  for (int i = 0; i < 1 * 2 * 3; i++) {
    EXPECT_EQ(et(i), i);
  }
  // TODO: check the content of et.
}

TEST(Eigen, Vector) {}

TEST(Eigen, Matrix) {}

}  // namespace framework
}  // namespace paddle
