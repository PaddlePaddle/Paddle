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

#include "paddle/phi/kernels/funcs/scatter.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

TEST(scatter, ScatterUpdate) {
  phi::DenseTensor src;
  phi::DenseTensor index;
  phi::DenseTensor output;

  auto* p_src = src.mutable_data<float>(phi::make_ddim({1, 4}),
                                        paddle::platform::CPUPlace());
  auto* p_index = index.mutable_data<int>(phi::make_ddim({1}),
                                          paddle::platform::CPUPlace());

  for (size_t i = 0; i < 4; ++i) {
    p_src[i] = static_cast<float>(i);
  }
  p_index[0] = 1;

  auto* p_output = output.mutable_data<float>(phi::make_ddim({4, 4}),
                                              paddle::platform::CPUPlace());

  for (int64_t i = 0; i < output.numel(); ++i) {
    p_output[i] = 0;
  }

  auto* cpu_place = new paddle::platform::CPUPlace();
  phi::CPUContext ctx(*cpu_place);
  phi::funcs::ScatterAssign<float>(ctx, src, index, &output);

  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(p_output[i], 0.0f);
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(output.data<float>()[i], 0.0f);
  for (size_t i = 4; i < 8; ++i) {
    EXPECT_EQ(p_output[i], static_cast<float>(i - 4));
  }
  for (size_t i = 4; i < 8; ++i)
    EXPECT_EQ(output.data<float>()[i], static_cast<float>(i - 4));
  for (size_t i = 8; i < 16; ++i) EXPECT_EQ(p_output[i], 0.0f);
  for (size_t i = 8; i < 16; ++i) EXPECT_EQ(output.data<float>()[i], 0.0f);

  delete cpu_place;
}
