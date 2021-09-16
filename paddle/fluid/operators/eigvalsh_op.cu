/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/eigvalsh_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename ValueType, typename T>
class EigvalshGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto input_var = ctx.Input<Tensor>("X");
    auto output_w_var = ctx.Output<Tensor>("Eigenvalues");
    auto output_v_var = ctx.Output<Tensor>("Eigenvectors");
    std::string lower = ctx.Attr<std::string>("UPLO");
    bool is_test = ctx.Attr<bool>("is_test");
    bool compute_vector = !is_test;
    bool is_lower = (lower == "L");
    math::MatrixEighFunctor<ValueType, T> functor;
    functor(ctx, *input_var, output_w_var, output_v_var, is_lower,
            compute_vector);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    eigvalsh, ops::EigvalshGPUKernel<float, float>,
    ops::EigvalshGPUKernel<double, double>,
    ops::EigvalshGPUKernel<float, paddle::platform::complex<float>>,
    ops::EigvalshGPUKernel<double, paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    eigvalsh_grad,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, float, float>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, double,
                            double>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, float,
                            paddle::platform::complex<float>>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, double,
                            paddle::platform::complex<double>>);
