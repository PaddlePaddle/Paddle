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

#include "paddle/fluid/operators/collective/c_barrier_op.h"
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CBarrierOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto place = ctx.GetPlace();
    int ring_id = ctx.Attr<int>("ring_id");
    auto comm =
        paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    auto dev_ctx = static_cast<platform::NPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = dev_ctx->stream();
    } else {
      stream = comm->stream();
    }

    VLOG(3) << "begin hccl barrier, parameter is: "
            << ", ring_id is " << ring_id;
    PADDLE_ENFORCE_NPU_SUCCESS(
        platform::dynload::HcclBarrier(comm->comm(), stream));
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with NPU."));
#endif
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(c_barrier, ops::CBarrierOpNPUKernel<int8_t>,
                       ops::CBarrierOpNPUKernel<int>,
                       ops::CBarrierOpNPUKernel<float>,
                       ops::CBarrierOpNPUKernel<plat::float16>);
