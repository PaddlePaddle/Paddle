// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/infrt/host_context/kernel_registry.h"

#include <gtest/gtest.h>

#include "paddle/infrt/host_context/kernel_utils.h"

namespace infrt::host_context {

int add_i32(int a, int b) { return a + b; }

TEST(KernelRegistry, basic) {
  KernelRegistry registry;
  std::string key = "infrt.test.add.i32";
  registry.AddKernel(key, INFRT_KERNEL(add_i32));

  auto* kernel_impl = registry.GetKernel(key);
  ASSERT_TRUE(kernel_impl);

  ValueRef a(1);
  ValueRef b(2);
  KernelFrameBuilder fbuilder;
  fbuilder.AddArgument(a.get());
  fbuilder.AddArgument(b.get());
  fbuilder.SetNumResults(1);

  kernel_impl(&fbuilder);

  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results[0]->get<int>(), 3);
}

}  // namespace infrt::host_context
