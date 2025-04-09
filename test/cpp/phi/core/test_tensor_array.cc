/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <sstream>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "paddle/common/errors.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_array.h"
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace tests {

using pstring = ::phi::dtype::pstring;

TEST(tensor_array, tensor_array_not_init) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LegacyLoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);
  DenseTensor tensor_0;
  tensor_0.set_meta(meta);

  std::vector<DenseTensor> tensors;
  tensors.push_back(tensor_0);
  tensors.push_back(tensor_0);
  tensors.push_back(tensor_0);

  TensorArray tensor_array(tensors);

  try {
    tensor_array.dims();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("dims") != std::string::npos);
  }

  try {
    tensor_array.place();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("place") != std::string::npos);
  }

  try {
    tensor_array.dtype();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("dtype") != std::string::npos);
  }

  try {
    tensor_array.layout();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("layout") != std::string::npos);
  }

  try {
    tensor_array.numel();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("numel") != std::string::npos);
  }

  try {
    tensor_array.valid();
  } catch (const common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("valid") != std::string::npos);
  }

  EXPECT_TRUE(!tensor_array.initialized());
}

TEST(tensor_array, tensor_array_init) {
  const DDim dims1({1, 2});
  const DDim dims2({1, 2, 3});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LegacyLoD lod{};

  DenseTensorMeta meta1(dtype, dims1, layout, lod);
  DenseTensorMeta meta2(dtype, dims2, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  DenseTensor tensor_0;
  tensor_0.set_meta(meta1);

  DenseTensor tensor_1;
  tensor_1.set_meta(meta2);

  std::vector<DenseTensor> tensors;
  tensors.push_back(tensor_0);
  tensors.push_back(tensor_1);
  tensors.push_back(tensor_0);

  TensorArray tensor_array(tensors);
  tensor_array.AllocateFrom(alloc, DataType::INT8);

  EXPECT_TRUE(tensor_array.initialized());
}

}  // namespace tests
}  // namespace phi
