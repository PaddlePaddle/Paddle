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

namespace infrt {
namespace host_context {

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

void TestFunc(const std::string& arg_0,
              const std::string& arg_1,
              const std::string& arg_2,
              Attribute<std::string> attr_0,
              Result<std::string> res_0,
              Result<std::string> res_1) {
  CHECK_EQ(arg_0, "arg_0");
  CHECK_EQ(arg_1, "arg_1");
  CHECK_EQ(arg_2, "arg_2");
  CHECK_EQ(attr_0.get(), "attr_0");

  // res_0.Set(Argument<std::string>(ValueRef(new Value())));
  // res_1.Set(Argument<std::string>(ValueRef(new Value())));
}

TEST(KernelRegistry, basic) {
  KernelFrameBuilder kernel_frame;

  Value arg_0(std::string{"arg_0"});
  Value arg_1(std::string{"arg_1"});
  Value arg_2(std::string{"arg_2"});
  Value attr_0(std::string{"attr_0"});

  kernel_frame.AddArgument(&arg_0);
  kernel_frame.AddArgument(&arg_1);
  kernel_frame.AddArgument(&arg_2);
  kernel_frame.AddAttribute(&attr_0);
  kernel_frame.SetNumResults(2);

  CHECK_EQ(kernel_frame.GetNumArgs(), 3);
  CHECK_EQ(kernel_frame.GetNumResults(), 2);
  CHECK_EQ(kernel_frame.GetNumAttributes(), 1);
  CHECK_EQ(kernel_frame.GetNumElements(), 6UL);

  CHECK_EQ(kernel_frame.GetArgAt<std::string>(2), "arg_2");
  CHECK_EQ(kernel_frame.GetAttributeAt(0)->get<std::string>(), "attr_0");

  KernelImpl<decltype(&TestFunc), TestFunc>::Invoke(&kernel_frame);
}

}  // namespace host_context
}  // namespace infrt
