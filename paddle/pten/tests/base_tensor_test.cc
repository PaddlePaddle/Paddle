/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/base_tensor.h"

#include <gtest/gtest.h>

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(BaseTensor, Constructor) {
  pt::TensorMeta meta(framework::make_ddim({5, 10}),
                      pt::Backend::kCPU,
                      pt::DataType::kFLOAT32,
                      pt::Layout::kNCHW,
                      0UL);
  pt::BaseTensor tensor(std::move(meta));
  ASSERT_EQ(tensor.dims().size(), 2);
  ASSERT_EQ(tensor.backend(), pt::Backend::kCPU);
  ASSERT_EQ(tensor.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(tensor.layout(), pt::Layout::kNCHW);
}

TEST(BaseTensor, Dims) {
  // impl later
}

TEST(BaseTensor, Place) {
  // impl later
}

TEST(BaseTensor, Data) {
  // impl later
}
