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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#include "paddle/fluid/operators/fused/xpu_fused_common_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedFeedForwardGradXPUKernel : public framework::OpKernel<T> {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    fused_feedforward_grad,
    ops::FusedFeedForwardGradXPUKernel<phi::XPUContext, float>,
    ops::FusedFeedForwardGradXPUKernel<phi::XPUContext,
                                       paddle::platform::float16>);

#endif
