/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/prelu.h"
#include "paddle/fluid/operators/prelu_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CUDAPReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());

    const T* alpha_ptr = alpha->data<T>();
    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    std::vector<int> input_shape = framework::vectorize2int(dim);

    if (mode == "channel") {
      math::PreluChannelWiseDirectCUDAFunctor<T> prelu_channel_wise;
      prelu_channel_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, input_shape);
    } else if (mode == "element") {
      math::PreluElementWiseDirectCUDAFunctor<T> prelu_element_wise;
      prelu_element_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, input_shape);
    } else {
      math::PreluScalarDirectCUDAFunctor<T> prelu_scalar;
      prelu_scalar(context.cuda_device_context().stream(), x_ptr, alpha_ptr,
                   o_ptr, input_shape);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    prelu, ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, double>);
