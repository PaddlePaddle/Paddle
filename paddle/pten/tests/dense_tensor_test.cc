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

#include "paddle/pten/core/dense_tensor.h"

#include <gtest/gtest.h>

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(DenseTensor, Constructor) {
  pten::DenseTensor tensor(pten::TensorMeta(framework::make_ddim({5, 10}),
                                            pten::Backend::CPU,
                                            pten::DataType::FLOAT32,
                                            pten::DataLayout::NCHW,
                                            0UL),
                           pten::TensorStatus());
  ASSERT_EQ(tensor.dims().size(), 2);
  ASSERT_EQ(tensor.backend(), pten::Backend::CPU);
  ASSERT_EQ(tensor.data_type(), pten::DataType::FLOAT32);
  ASSERT_EQ(tensor.layout(), pten::DataLayout::NCHW);
}

TEST(DenseTensor, Dims) {
  // impl later
}

TEST(DenseTensor, Place) {
  // impl later
}

TEST(DenseTensor, Data) {
  // impl later
}
