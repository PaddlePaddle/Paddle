// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/context.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

#ifdef LITE_WITH_X86
TEST(ContextScheduler, NewContext) {
  auto ctx1_p = ContextScheduler::Global().NewContext(TargetType::kX86);
  auto ctx2_p = ContextScheduler::Global().NewContext(TargetType::kX86);
  ASSERT_FALSE(ctx1_p.get() == ctx2_p.get());

  auto& ctx1 = ctx1_p->As<X86Context>();
  auto& ctx2 = ctx2_p->As<X86Context>();

  ASSERT_EQ(ctx1.name(), "X86Context");
  ASSERT_EQ(ctx2.name(), "X86Context");

  ASSERT_FALSE(ctx1.x86_device_context() == nullptr ||
               ctx2.x86_device_context() == nullptr);
  ASSERT_FALSE(ctx1.x86_execution_context() == nullptr ||
               ctx2.x86_execution_context() == nullptr);

  ASSERT_TRUE(ctx1.x86_device_context() != ctx2.x86_device_context());
  ASSERT_TRUE(ctx1.x86_execution_context() != ctx2.x86_execution_context());

  using device_ctx_t = ::paddle::platform::CPUDeviceContext;
  using exec_ctx_t = ::paddle::framework::ExecutionContext;
  auto* device_ctx = new device_ctx_t;
  ctx1.SetX86DeviceContext(std::unique_ptr<device_ctx_t>(device_ctx));
  ctx1.SetX86ExecutionContext(
      std::unique_ptr<exec_ctx_t>(new exec_ctx_t(*device_ctx)));
}
#endif

}  // namespace lite
}  // namespace paddle
