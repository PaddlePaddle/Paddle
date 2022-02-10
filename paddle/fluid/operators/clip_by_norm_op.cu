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

#include "paddle/fluid/operators/clip_by_norm_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <>
class ClipByNormKernel<platform::CUDADeviceContext, platform::float16>
    : public framework::OpKernel<platform::float16> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<float>("max_norm");
    auto in_var = context.InputVar("X");
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    Tensor* output = nullptr;
    const Tensor* input = nullptr;
    if (in_var->IsType<framework::LoDTensor>()) {
      input = context.Input<Tensor>("X");

      output = context.Output<Tensor>("Out");
      output->mutable_data<platform::float16>(context.GetPlace());
    } else if (in_var->IsType<pten::SelectedRows>()) {
      auto* x = context.Input<pten::SelectedRows>("X");

      // merge ids in selected rows first
      math::scatter::MergeAdd<platform::CUDADeviceContext, platform::float16>
          merge_func;
      pten::SelectedRows* merged_input =
          const_cast<framework::Scope&>(context.scope())
              .Var()
              ->GetMutable<pten::SelectedRows>();
      merge_func(context.template device_context<platform::CUDADeviceContext>(),
                 *x, merged_input);
      input = &(merged_input->value());

      pten::SelectedRows* output_selected_rows =
          context.Output<pten::SelectedRows>("Out");
      output_selected_rows->set_rows(merged_input->rows());
      output_selected_rows->set_height(merged_input->height());
      output = output_selected_rows->mutable_value();
      output->Resize(merged_input->value().dims());
      output->mutable_data<platform::float16>(context.GetPlace());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid input variable type, only support LodTensor and "
          "SelectedRows types, but got type is %s.",
          framework::ToTypeName(in_var->Type())));
    }

    PADDLE_ENFORCE_NOT_NULL(input,
                            platform::errors::InvalidArgument(
                                "Input(X) of ClipByNormOp should not be null. "
                                "Please check if it is created correctly."));
    std::vector<int> reduce_dims;
    reduce_dims.resize(input->dims().size());
    for (int i = 0; i < reduce_dims.size(); ++i) {
      reduce_dims[i] = i;
    }
    Tensor tmp = context.AllocateTmpTensor<float, platform::CUDADeviceContext>(
        {1}, dev_ctx);
    TensorReduceImpl<platform::float16, float, kps::AddFunctor,
                     kps::SquareFunctor<platform::float16, float>>(
        dev_ctx, *input, &tmp, kps::SquareFunctor<platform::float16, float>(),
        reduce_dims, dev_ctx.stream());
    auto tmp_eigen = EigenVector<float>::Flatten(tmp);
    auto x_norm = tmp_eigen.sqrt();

    auto x = EigenVector<platform::float16>::Flatten(*input);
    auto out = EigenVector<platform::float16>::Flatten(*output);

    auto& place =
        *context.template device_context<platform::CUDADeviceContext>()
             .eigen_device();

    auto temp = (x_norm <= max_norm).template cast<float>();
    auto epsilon =
        ((x_norm <= static_cast<float>(1e-30)).all().template cast<float>()) *
        static_cast<float>(1e-6);

    auto scaling =
        (temp + (static_cast<float>(1) - temp) * max_norm / (x_norm + epsilon))
            .template cast<platform::float16>();
    Eigen::array<int, 1> one_dim{{1}};
    Eigen::DSizes<int, 1> m_dsize(input->numel());

    out.device(place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    clip_by_norm,
    ops::ClipByNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ClipByNormKernel<paddle::platform::CUDADeviceContext, plat::float16>);
