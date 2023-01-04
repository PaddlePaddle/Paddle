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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = phi::DenseTensor;

template <typename T>
class SequenceSoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto& lod = x->lod();
    auto& dims = x->dims();

    const size_t level = lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        dims[0],
        static_cast<int64_t>(lod[level].back()),
        platform::errors::InvalidArgument(
            "The first dimension of Input(X) should be equal to the sum of all "
            "sequences' lengths. But received first dimension of Input(X) is "
            "%d, the sum of all sequences' lengths is %d.",
            dims[0],
            static_cast<int64_t>(lod[level].back())));
    PADDLE_ENFORCE_EQ(dims[0],
                      x->numel(),
                      platform::errors::InvalidArgument(
                          "The width of each timestep in Input(X) of "
                          "SequenceSoftmaxOp should be 1."));

    out->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);
      Tensor x_i = x->Slice(start_pos, end_pos);
      Tensor out_i = out->Slice(start_pos, end_pos);

      // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos - start_pos)
      framework::DDim dims_i =
          // phi::make_ddim({1UL, end_pos - start_pos, 1UL, 1UL});
          phi::make_ddim({1UL, end_pos - start_pos});
      x_i.Resize(dims_i);
      out_i.Resize(dims_i);
      phi::funcs::SoftmaxCUDNNFunctor<T, phi::GPUContext>()(
          ctx.template device_context<phi::GPUContext>(), &x_i, &out_i);
    }
  }
};

template <typename T>
class SequenceSoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<LoDTensor>("Out");
    auto* out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<LoDTensor>("X");
    auto* x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    if (x_grad) {
      x_grad->set_lod(x->lod());
    }
    auto& lod = x->lod();
    const size_t level = lod.size() - 1;

    x_grad->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);

      Tensor out_i = out->Slice(start_pos, end_pos);
      Tensor out_grad_i = out_grad->Slice(start_pos, end_pos);
      Tensor x_grad_i = x_grad->Slice(start_pos, end_pos);

      // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos - start_pos)
      framework::DDim dims_i = phi::make_ddim({1UL, end_pos - start_pos});
      out_i.Resize(dims_i);
      out_grad_i.Resize(dims_i);
      x_grad_i.Resize(dims_i);
      phi::funcs::SoftmaxGradCUDNNFunctor<T, phi::GPUContext>()(
          ctx.template device_context<phi::GPUContext>(),
          &out_i,
          &out_grad_i,
          &x_grad_i);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#ifdef PADDLE_WITH_HIP
// MIOPEN not support float64
REGISTER_OP_KERNEL(sequence_softmax,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxCUDNNKernel<float>);
REGISTER_OP_KERNEL(sequence_softmax_grad,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxGradCUDNNKernel<float>);
#else
REGISTER_OP_KERNEL(sequence_softmax,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxCUDNNKernel<float>,
                   ops::SequenceSoftmaxCUDNNKernel<double>);
REGISTER_OP_KERNEL(sequence_softmax_grad,
                   CUDNN,
                   ::paddle::platform::CUDAPlace,
                   ops::SequenceSoftmaxGradCUDNNKernel<float>,
                   ops::SequenceSoftmaxGradCUDNNKernel<double>);
#endif
