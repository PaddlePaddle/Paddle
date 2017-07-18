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

#include "paddle/framework/tensor.h"

TEST(Eigen, Tensor) {
  using paddle::platform::Tensor;
  using paddle::platform::EigenTensor;
  using paddle::platform::make_ddim;

  Tensor t;
  float* p = t.mutable_data<float>(make_ddim({1, 2, 3}), CPUPlace());
  for (int i = 0; i < 1 * 2 * 3; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenTensor::Type et = EigenTensor::From(t);
  // TODO: check the content of et.
}

TEST(Eigen, Vector) {}

TEST(Eigen, Matrix) {}
