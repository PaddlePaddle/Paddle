//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/operators/one_hot_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class OneHotXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int depth = context.Attr<int>("depth");
    if (context.HasInput("depth_tensor")) {
      auto* depth_tensor = context.Input<Tensor>("depth_tensor");
      auto* depth_data = depth_tensor->data<int32_t>();
      if (platform::is_xpu_place(depth_tensor->place())) {
        xpu_memcpy(static_cast<void*>(&depth),
                   static_cast<const void*>(depth_data), sizeof(int32_t),
                   XPU_DEVICE_TO_HOST);
      } else {
        depth = depth_data[0];
      }
      auto in_dims = in->dims();
      framework::DDim out_dims(in_dims);
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int len = in->numel();
    int ret = xpu::one_hot<T>(dev_ctx.x_context(), in->data<T>(),
                              out->mutable_data<float>(context.GetPlace()), len,
                              depth);

    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU one_hot kernel return wrong value[%d %s]", ret,
                          XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    one_hot, ops::OneHotXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::OneHotXPUKernel<paddle::platform::XPUDeviceContext, int64_t>);
#endif
