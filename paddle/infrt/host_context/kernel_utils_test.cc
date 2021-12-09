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

#include "paddle/infrt/host_context/kernel_utils.h"

#include <gtest/gtest.h>

namespace infrt::host_context {

int add_i32(int a, int b) { return a + b; }
float add_f32(float a, float b) { return a + b; }
std::pair<int, float> add_pair(int a, float b) { return {a, b}; }

TEST(KernelImpl, i32) {
  KernelFrameBuilder fbuilder;
  ValueRef a(new Value(1));
  ValueRef b(new Value(2));
  fbuilder.AddArgument(a.get());
  fbuilder.AddArgument(b.get());
  fbuilder.SetNumResults(1);

  INFRT_KERNEL(add_i32)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results.front()->get<int>(), 3);
}

TEST(KernelImpl, f32) {
  KernelFrameBuilder fbuilder;
  ValueRef a(new Value(1.f));
  ValueRef b(new Value(2.f));
  fbuilder.AddArgument(a.get());
  fbuilder.AddArgument(b.get());
  fbuilder.SetNumResults(1);

  INFRT_KERNEL(add_f32)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results.front()->get<float>(), 3.f);
}

TEST(KernelImpl, pair) {
  KernelFrameBuilder fbuilder;
  ValueRef a(new Value(1));
  ValueRef b(new Value(3.f));

  fbuilder.AddArgument(a.get());
  fbuilder.AddArgument(b.get());
  fbuilder.SetNumResults(2);

  INFRT_KERNEL(add_pair)(&fbuilder);
  auto results = fbuilder.GetResults();
  ASSERT_EQ(results.size(), 2UL);
  ASSERT_EQ(results[0]->get<int>(), 1);
  ASSERT_EQ(results[1]->get<float>(), 3.f);
}

}  // namespace infrt::host_context
