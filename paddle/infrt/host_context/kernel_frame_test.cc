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

#include <gtest/gtest.h>

#include "paddle/infrt/host_context/kernel_frame.h"

namespace infrt {
namespace host_context {
/*
TEST(KernelRegistry, basic) {
  KernelFrameBuilder kernel_frame;

  Value arg_0(std::string{"arg_0"});
  Value arg_1(std::string{"arg_1"});
  Value arg_2(std::string{"arg_2"});
  Value res_0(std::string{"res_0"});
  Value res_1(std::string{"res_1"});
  Value attr_0(std::string{"attr_0"});

  kernel_frame.AddArgument(&arg_0);
  kernel_frame.AddArgument(&arg_1);
  kernel_frame.AddArgument(&arg_2);
  kernel_frame.SetResults({&res_0, &res_1});
  kernel_frame.AddAttribute(&attr_0);

  CHECK_EQ(kernel_frame.GetNumArgs(), 3);
  CHECK_EQ(kernel_frame.GetNumResults(), 2);
  CHECK_EQ(kernel_frame.GetNumAttributes(), 1);
  CHECK_EQ(kernel_frame.GetNumElements(), 6UL);

  CHECK_EQ(kernel_frame.GetArgAt<std::string>(2), "arg_2");
  CHECK_EQ(kernel_frame.GetAttributeAt(0)->get<std::string>(), "attr_0");
  CHECK_EQ(kernel_frame.GetResults()[1]->get<std::string>(), "res_1");
}
*/

TEST(KernelRegistry, basic) {
  KernelFrameBuilder kernel_frame;

  Value arg_0(std::string{"arg_0"});
  Value arg_1(std::string{"arg_1"});
  Value arg_2(std::string{"arg_2"});
  Value attr_0(std::string{"attr_0"});
  Value res_0(std::string{"res_0"});
  Value res_1(std::string{"res_1"});

  kernel_frame.AddArgument(&arg_0);
  kernel_frame.AddArgument(&arg_1);
  kernel_frame.AddArgument(&arg_2);
  kernel_frame.AddAttribute(&attr_0);
  kernel_frame.SetResults({&res_0, &res_1});

  CHECK_EQ(kernel_frame.GetNumArgs(), 3);
  CHECK_EQ(kernel_frame.GetNumResults(), 2);
  CHECK_EQ(kernel_frame.GetNumAttributes(), 1);
  CHECK_EQ(kernel_frame.GetNumElements(), 6UL);

  CHECK_EQ(kernel_frame.GetArgAt<std::string>(2), "arg_2");
  CHECK_EQ(kernel_frame.GetAttributeAt(0)->get<std::string>(), "attr_0");
  CHECK_EQ(kernel_frame.GetResults()[1]->get<std::string>(), "res_1");
}

}  // namespace host_context
}  // namespace infrt
