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

#include "paddle/operators/scatter.h"
#include "paddle/framework/ddim.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/place.h"

#include <gtest/gtest.h>
#include <iostream>
#include <string>

TEST(scatter, ScatterUpdate) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  using namespace paddle::operators;

  Tensor* src = new Tensor();
  Tensor* index = new Tensor();
  Tensor* output = new Tensor();

  float* p_src = nullptr;
  int* p_index = nullptr;
  p_src = src->mutable_data<float>(make_ddim({1, 4}), CPUPlace());
  p_index = index->mutable_data<int>(make_ddim({1}), CPUPlace());

  for (size_t i = 0; i < 4; ++i) p_src[i] = float(i);
  p_index[0] = 1;

  float* p_output = output->mutable_data<float>(make_ddim({4, 4}), CPUPlace());

  auto* cpu_place = new paddle::platform::CPUPlace();
  paddle::platform::CPUDeviceContext ctx(*cpu_place);
  ScatterAssign<float>(ctx, *src, *index, output);

  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(p_output[i], float(0));
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(output->data<float>()[i], float(0));
  for (size_t i = 4; i < 8; ++i) EXPECT_EQ(p_output[i], float(i - 4));
  for (size_t i = 4; i < 8; ++i)
    EXPECT_EQ(output->data<float>()[i], float(i - 4));
  for (size_t i = 8; i < 16; ++i) EXPECT_EQ(p_output[i], float(0));
  for (size_t i = 8; i < 16; ++i) EXPECT_EQ(output->data<float>()[i], float(0));

  delete src;
  delete index;
  delete output;
}
