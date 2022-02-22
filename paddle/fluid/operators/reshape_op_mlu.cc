/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class Reshape2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");

    std::vector<int32_t> target_shape_vector;
    auto shape_tensor_vector = ctx.MultiInput<framework::Tensor>("ShapeTensor");
    if (shape_tensor_vector.size() > 0) {
      for (auto* shape_tensor : shape_tensor_vector) {
        PADDLE_ENFORCE_EQ(
            shape_tensor->dims().size(), 1,
            platform::errors::InvalidArgument(
                "If the element type of 'shape' in Reshape Op is Tensor, "
                "the element's shape must be [1]. But received the element's "
                "shape is [%d]",
                shape_tensor->dims().size()));

        target_shape_vector.push_back(GetDataFromTensor<int>(shape_tensor)[0]);
      }
    } else {
      auto* shape_tensor = ctx.HasInput("Shape")
                               ? ctx.Input<framework::LoDTensor>("Shape")
                               : nullptr;
      if (shape_tensor) {
        target_shape_vector = GetDataFromTensor<int>(shape_tensor);
      } else {
        target_shape_vector = ctx.Attr<std::vector<int>>("shape");
        PADDLE_ENFORCE_GT(
            target_shape_vector.size(), 0,
            platform::errors::InvalidArgument(
                "The length of shape attribute should be larger than 0 when "
                "input ShapeTensor and Shape are empty!"));
      }
    }

    int num_negative =
        std::count(target_shape_vector.begin(), target_shape_vector.end(), -1);
    PADDLE_ENFORCE_LE(
        num_negative, 1,
        platform::errors::InvalidArgument(
            "The max number of -1 in shape attribute or shape tensor is 1 "
            "but received %d.",
            num_negative));
    auto it_zero =
        std::find(target_shape_vector.begin(), target_shape_vector.end(), 0);
    if (it_zero != target_shape_vector.end()) {
      int x_rank = x->dims().size();
      for (size_t i = 0; i < target_shape_vector.size(); i++) {
        if (target_shape_vector[i] == 0) {
          PADDLE_ENFORCE_LT(
              i, x_rank,
              platform::errors::InvalidArgument(
                  "The index of 0 in shape attribute or shape tensor",
                  "should be less than input dim size, ",
                  "but the index is %d and input dim size is %d", i, x_rank));
          target_shape_vector[i] = x->dims().at(i);
        }
      }
    }

    auto it =
        std::find(target_shape_vector.begin(), target_shape_vector.end(), -1);
    if (it != target_shape_vector.end()) {
      auto ddim_out_vec = phi::vectorize(x->dims());
      int ddim_out_product = std::accumulate(
          ddim_out_vec.begin(), ddim_out_vec.end(), 1, std::multiplies<int>());
      int reshape_out_product = std::accumulate(target_shape_vector.begin(),
                                                target_shape_vector.end(), -1,
                                                std::multiplies<int>());
      int index = std::distance(target_shape_vector.begin(), it);
      target_shape_vector[index] = ddim_out_product / reshape_out_product;
    }

    auto out_dims = phi::make_ddim(target_shape_vector);
    out->mutable_data<T>(out_dims, ctx.GetPlace());

    // output should copy to mlu
    framework::TensorCopy(
        *x, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class Reshape2GradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto in_dims = d_x->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(in_dims);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    reshape2, ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, int64_t>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, bool>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::Reshape2MLUKernel<paddle::platform::MLUDeviceContext,
                           paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    reshape2_grad,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, int64_t>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, bool>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::Reshape2GradMLUKernel<paddle::platform::MLUDeviceContext,
                               paddle::platform::float16>);
