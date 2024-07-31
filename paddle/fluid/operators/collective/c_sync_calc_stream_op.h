/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class CSyncCalcStreamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

template <typename T, typename DeviceContext>
class CSyncCalcStreamKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)

    auto place = ctx.GetPlace();
    auto dev_ctx = static_cast<phi::GPUContext*>(
        phi::DeviceContextPool::Instance().Get(place));

    phi::backends::gpu::GpuStreamSync(dev_ctx->stream());

#elif defined(PADDLE_WITH_XPU_BKCL)
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::XPU,
                      true,
                      common::errors::PreconditionNotMet(
                          "Sync stream op can run on xpu place only for now."));

    auto dev_ctx = static_cast<phi::XPUContext*>(
        phi::DeviceContextPool::Instance().Get(place));
    dev_ctx->Wait();
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
