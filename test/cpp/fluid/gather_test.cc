/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/gather.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/common/place.h"

TEST(Gather, GatherData) {
  phi::DenseTensor* src = new phi::DenseTensor();
  phi::DenseTensor* index = new phi::DenseTensor();
  phi::DenseTensor* output = new phi::DenseTensor();

  int* p_src = nullptr;
  int* p_index = nullptr;
  p_src = src->mutable_data<int>(common::make_ddim({3, 4}), phi::CPUPlace());
  p_index = index->mutable_data<int>(common::make_ddim({2}), phi::CPUPlace());

  for (int i = 0; i < 12; ++i) p_src[i] = i;
  p_index[0] = 1;
  p_index[1] = 0;

  int* p_output =
      output->mutable_data<int>(common::make_ddim({2, 4}), phi::CPUPlace());

  auto* cpu_place = new phi::CPUPlace();
  phi::CPUContext ctx(*cpu_place);
  phi::funcs::CPUGather<int>(ctx, *src, *index, output);
  delete cpu_place;
  cpu_place = nullptr;
  for (int i = 0; i < 4; ++i) EXPECT_EQ(p_output[i], i + 4);
  for (int i = 4; i < 8; ++i) EXPECT_EQ(p_output[i], i - 4);

  delete src;
  delete index;
  delete output;
}
