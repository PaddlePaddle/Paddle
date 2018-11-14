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
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/modified_huber_loss_op.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

struct ModifiedHuberLossBackward {
  template <typename Tuple>
  HOSTDEVICE void operator()(Tuple t) const {
    auto inter_val = thrust::get<1>(t);
    auto y_val = thrust::get<2>(t);
    auto out_grad = thrust::get<3>(t);
    if (inter_val < -1) {
      thrust::get<0>(t) = -4 * (2 * y_val - 1) * out_grad;
    } else if (inter_val < 1) {
      thrust::get<0>(t) = -2 * (1 - inter_val) * (2 * y_val - 1) * out_grad;
    } else {
      thrust::get<0>(t) = 0;
    }
  }
};

template <typename T>
class ModifiedHuberLossGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("Y");
    auto* in1 = context.Input<Tensor>("IntermediateVal");
    auto* in2 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));

    if (out0) {
      auto counts = framework::product(in1->dims());
      auto y_ptr = thrust::device_pointer_cast(in0->data<T>());
      auto inter_val_ptr = thrust::device_pointer_cast(in1->data<T>());
      auto out_grad_ptr = thrust::device_pointer_cast(in2->data<T>());
      thrust::device_ptr<T> x_grad_ptr(
          out0->mutable_data<T>(context.GetPlace()));

      auto iter_begin = thrust::make_zip_iterator(
          thrust::make_tuple(x_grad_ptr, inter_val_ptr, y_ptr, out_grad_ptr));

      auto iter_end = thrust::make_zip_iterator(
          thrust::make_tuple(x_grad_ptr + counts, inter_val_ptr + counts,
                             y_ptr + counts, out_grad_ptr + counts));

      thrust::for_each(iter_begin, iter_end, ModifiedHuberLossBackward());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    modified_huber_loss,
    ops::ModifiedHuberLossKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(modified_huber_loss_grad,
                        ops::ModifiedHuberLossGradGPUKernel<float>);
