/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_call_stack.h"

#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

static void ThrowEnforceNotMet() {
  PADDLE_THROW(
      common::errors::InvalidArgument("\n----------------------\nError Message "
                                      "Summary:\n----------------------\n"
                                      "Created error."));
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

TEST(OpCallStack, InsertCallStackInfo) {
  try {
    paddle::framework::details::ThrowEnforceNotMet();
  } catch (paddle::platform::EnforceNotMet &exception) {
    paddle::framework::AttributeMap attr_map;
    std::string stack_test_str = "test for op callstack";
    std::vector<std::string> stack_test_vec;
    stack_test_vec.emplace_back(stack_test_str);
    attr_map["op_callstack"] = stack_test_vec;
    paddle::framework::InsertCallStackInfo("test", attr_map, &exception);
    paddle::framework::InsertCallStackInfo("test", stack_test_vec, &exception);
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find(stack_test_str) != std::string::npos);
    EXPECT_TRUE(ex_msg.find("[operator < test > error]") != std::string::npos);
  }
}

TEST(OpCallStack, AppendErrorOpHint) {
  try {
    paddle::framework::details::ThrowEnforceNotMet();
  } catch (paddle::platform::EnforceNotMet &exception) {
    paddle::framework::AppendErrorOpHint("test", &exception);
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find("[operator < test > error]") != std::string::npos);
  }
}
