// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LookupTableKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* table = context.Input<phi::DenseTensor>("W");
    auto* ids = context.Input<phi::DenseTensor>("Ids");
    auto* out = context.Output<phi::DenseTensor>("Out");
    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    int xm = table->dims()[0];
    int n = table->dims()[1];
    int ym = ids->numel();

    int r = xpu::embedding<T, int64_t>(dev_ctx.x_context(),
                                       table->data<T>(),
                                       ids->data<int64_t>(),
                                       out->data<T>(),
                                       xm,
                                       n,
                                       ym,
                                       padding_idx);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    lookup_table,
    ops::LookupTableKernel<paddle::platform::XPUDeviceContext, float>);

#endif
