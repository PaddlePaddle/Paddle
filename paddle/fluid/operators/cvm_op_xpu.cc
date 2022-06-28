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

#ifdef PADDLE_WITH_XPU

//#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cvm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class CVMXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<LoDTensor>("X");
    const T* x_data = x->data<T>();
    auto batch_size = x->dims()[0];
    auto numel = x->numel();
    //auto item_size = numel / batch_size;
    auto use_cvm = context.Attr<bool>("use_cvm");
    auto* y = context.Output<LoDTensor>("Y");
    T* y_data = y->mutable_data<T>(context.GetPlace());

    // for Input X do not have Lod Information.
    auto xpu_context =
        context.template device_context<DeviceContext>().x_context();

    //cvm(Context *ctx, const float *x, float *y, int batch_size, int len, bool use_cvm);

    int r = xpu::cvm<T>(xpu_context, x_data, y_data, batch_size, numel, use_cvm);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                platform::errors::External(
                               "The cvm XPU OP return wrong value[%d %s]",
                               r, XPUAPIErrorMsg[r]));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(cvm, ops::CVMXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
