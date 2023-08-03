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

#include "paddle/fluid/operators/expand_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ExpandXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* out = context.Output<phi::DenseTensor>("Out");

    auto in_dims = x->dims();
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    auto expand_shape = get_expand_times(context);
    auto vec_in_dims = phi::vectorize<int>(in_dims);

    framework::DDim out_dims(in_dims);
    for (size_t i = 0; i < expand_shape.size(); ++i) {
      out_dims[i] *= expand_shape[i];
    }

    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    auto& x_shape = vec_in_dims;
    auto out_shape = phi::vectorize<int>(out_dims);
    int r = XPU_SUCCESS;
    if (std::is_same<T, bool>::value) {
      auto x_data = reinterpret_cast<const int8_t*>(x->data<T>());
      auto out_data = reinterpret_cast<int8_t*>(out->data<T>());
      r = xpu::broadcast<int8_t>(
          dev_ctx.x_context(), x_data, out_data, x_shape, out_shape);
    } else {
      auto x_data = reinterpret_cast<const XPUType*>(x->data<T>());
      auto out_data = reinterpret_cast<XPUType*>(out->data<T>());
      r = xpu::broadcast<XPUType>(
          dev_ctx.x_context(), x_data, out_data, x_shape, out_shape);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    expand,
    ops::ExpandXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::ExpandXPUKernel<paddle::platform::XPUDeviceContext,
                         paddle::platform::float16>,
    ops::ExpandXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::ExpandXPUKernel<paddle::platform::XPUDeviceContext, int64_t>,
    ops::ExpandXPUKernel<paddle::platform::XPUDeviceContext, bool>);
#endif
