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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/core/dist_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace tests {

TEST(dist_tensor, basic) {
  DistTensor dist_tensor;
  // test meta
  EXPECT_EQ(dist_tensor.numel(), int64_t(0));
  MetaTensor meta_tensor(&dist_tensor);
  meta_tensor.set_layout(phi::DataLayout::NCHW);
  EXPECT_EQ(meta_tensor.layout(), phi::DataLayout::NCHW);
  EXPECT_TRUE(dist_tensor.valid());
  auto meta = dist_tensor.meta();
  EXPECT_NO_THROW({ dist_tensor.set_meta(meta); });
  EXPECT_THROW({ dist_tensor.set_meta(std::move(meta)); },
               phi::enforce::EnforceNotMet);
  // test allocate
  EXPECT_THROW({ dist_tensor.AllocateFrom(nullptr, DataType::INT8, 0, true); },
               phi::enforce::EnforceNotMet);
  // dist attr is not assigned yet
  EXPECT_TRUE(dist_tensor.get_dist_attr() == nullptr);
}
}  // namespace tests
}  // namespace phi
