// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>

#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class ReduceProdXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("This kernel only runs on XPU."));
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    std::vector<int> xdims;
    for (int i = 0; i < input->dims().size(); i++) {
      xdims.push_back(input->dims()[i]);
    }
    auto rdims = context.Attr<std::vector<int>>("dim");
    const auto& input_dim_size = input->dims().size();

    std::vector<int> reduce_dims;
    if (reduce_all) {
      for (size_t i = 0; i < xdims.size(); i++) {
        reduce_dims.push_back(static_cast<int>(i));
      }
    } else {
      for (size_t i = 0; i < rdims.size(); ++i) {
        if (rdims[i] < 0) {
          reduce_dims.push_back(rdims[i] + input_dim_size);
        } else {
          reduce_dims.push_back(rdims[i]);
        }
      }
    }
    int r = xpu::reduce_prod(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(input->data<T>()),
        reinterpret_cast<XPUType*>(output->data<T>()), xdims, reduce_dims);

    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU reduce_prod kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(
    reduce_prod,
    ops::ReduceProdXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
