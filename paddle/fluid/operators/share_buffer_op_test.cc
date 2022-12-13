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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(share_buffer);

PD_DECLARE_KERNEL(share_buffer, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(share_buffer, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace framework {

TEST(test_share_buffer_op, test_share_buffer_op) {
  std::vector<std::string> inputs = {"X1", "X2"};
  std::vector<std::string> outputs = {"Y1", "Y2"};
  std::vector<DDim> dims = {{2, 3, 4}, {5, 6}};
  std::vector<bool> share_dims_and_dtype = {false, true};

  size_t n = inputs.size();
  EXPECT_EQ(n, outputs.size());
  EXPECT_EQ(n, dims.size());
  EXPECT_EQ(n, share_dims_and_dtype.size());

  OpDesc desc;
  desc.SetType("share_buffer");
  desc.SetInput("X", inputs);
  desc.SetOutput("Out", outputs);
  desc.SetOutput("XOut", inputs);
  desc.SetAttr("share_dims_and_dtype", share_dims_and_dtype);

  auto op = OpRegistry::CreateOp(desc);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::Place place = platform::CUDAPlace(0);
#else
  platform::Place place = platform::CPUPlace();
#endif

  Scope scope;
  for (size_t i = 0; i < n; ++i) {
    auto *in_tensor = scope.Var(inputs[i])->GetMutable<phi::DenseTensor>();
    in_tensor->Resize(dims[i]);
    in_tensor->mutable_data<float>(place);
    scope.Var(outputs[i])->GetMutable<phi::DenseTensor>();
  }
  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  for (size_t i = 0; i < n; ++i) {
    const auto &in_tensor = scope.Var(inputs[i])->Get<phi::DenseTensor>();
    const auto &out_tensor = scope.Var(outputs[i])->Get<phi::DenseTensor>();
    EXPECT_TRUE(out_tensor.IsSharedBufferWith(in_tensor));
  }
}

}  // namespace framework
}  // namespace paddle
