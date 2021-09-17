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
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/squared_l2_norm_op.h"

namespace ops = paddle::operators;

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SquaredL2NormCudaKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    std::vector<int> reduce_dims;
    reduce_dims.resize(x->dims().size());
    for (int i = 0; i < reduce_dims.size(); ++i) {
      reduce_dims[i] = i;
    }

    TensorReduceFunctorImpl<T, T, SquareSum>(*x, out, reduce_dims,
                                             dev_ctx.stream());
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    squared_l2_norm,
    ops::SquaredL2NormCudaKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SquaredL2NormCudaKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    squared_l2_norm_grad,
    ops::SquaredL2NormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SquaredL2NormGradKernel<paddle::platform::CUDADeviceContext, double>);
