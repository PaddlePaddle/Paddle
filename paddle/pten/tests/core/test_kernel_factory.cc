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

#include <iostream>
#include <sstream>

#include "paddle/pten/core/kernel_factory.h"
#include "paddle/pten/core/kernel_registry.h"

#include "gtest/gtest.h"

PT_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

namespace pten {
namespace tests {

// TODO(chenweihang): add more unittests later

TEST(KernelKey, ConstructAndOStream) {
  pten::KernelKey key(
      pten::Backend::CPU, pten::DataLayout::NCHW, pten::DataType::FLOAT32);
  EXPECT_EQ(key.backend(), pten::Backend::CPU);
  EXPECT_EQ(key.layout(), pten::DataLayout::NCHW);
  EXPECT_EQ(key.dtype(), pten::DataType::FLOAT32);
  std::ostringstream oss;
  oss << key;
  std::cout << oss.str();
  oss.flush();
}

TEST(KernelFactory, SelectedKernelMap) {
  auto kernel_map = pten::KernelFactory::Instance().SelectKernelMap("scale");
  EXPECT_GT(kernel_map.size(), 1UL);
  for (auto& iter : kernel_map) {
    std::cout << iter.first << ": " << iter.second;
  }
}

}  // namespace tests
}  // namespace pten
