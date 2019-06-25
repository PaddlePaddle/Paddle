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

#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

TEST(elementwise_add, init) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "elementwise_add", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";

  lite::Tensor X, Y, Out;
  operators::ElementwiseParam param;
  param.X = &X;
  param.Y = &Y;
  param.Out = &Out;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenClContext>().InitOnce();

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  X.Resize({4, 3, 10, 10});
  Y.Resize({4, 3, 10, 10});
  Out.Resize({4, 3, 10, 10});

  auto* x_data = X.mutable_data<float>();
  auto* y_data = Y.mutable_data<float>();
  auto* out_data = Out.mutable_data<float>();

  for (int i = 0; i < 4 * 3 * 10 * 10; i++) {
    x_data[i] = 1.1 * i;
    y_data[i] = 2.3 * i;
  }

  kernel->Launch();

  for (int i = 0; i < 4 * 3 * 10 * 10; i++) {
    EXPECT_NEAR(out_data[i], static_cast<float>(3.4 * i), 1e-6);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kNCHW, def);
