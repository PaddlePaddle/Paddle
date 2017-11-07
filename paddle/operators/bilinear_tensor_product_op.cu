/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include "paddle/operators/bilinear_tensor_product_op.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class BilinearTensorProductCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto y_mat = EigenMatrix<T>::From(*y);
    auto batch_size = x->dims()[0];
    auto weight_dims = weight->dims();

    auto place = ctx.GetEigenDevice<Place>();
    auto cpu_place = ctx.GetEigenDevice<platform::CPUPlace>();

    // Copy the output to cpu.
    Tensor output_cpu;
    output_cpu.CopyFrom(*out, platform::CPUPlace(), ctx.device_context());
    auto* output_cpu_ptr = output_cpu.data<T>();
    auto output_cpu_mat = EigenMatrix<T>::From(output_cpu);

    // Create the temporary variables.
    Tensor left_mul;
    left_mul.mutable_data<T>(framework::make_ddim({batch_size, weight_dims[2]}),
                             ctx.GetPlace());
    auto left_mul_mat = EigenMatrix<T>::From(left_mul);
    Tensor output_col;
    output_col.mutable_data<T>(framework::make_ddim({batch_size}),
                               ctx.GetPlace());
    auto output_col_vec = EigenVector<T>::From(output_col);

    for (size_t i = 0; i < weight_dims[0]; ++i) {
      Tensor weight_mat = weight->Slice(i, i + 1).Resize(
          framework::make_ddim({weight_dims[1], weight_dims[2]}));
      math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans,
                           batch_size, weight_dims[2], weight_dims[1], 1,
                           x->data<T>(), weight_mat.data<T>(), 0,
                           left_mul.data<T>());
      output_col_vec.device(place) =
          (left_mul_mat * y_mat).sum(Eigen::DSizes<int, 1>(1));

      // Copy the output_col to cpu.
      Tensor output_col_cpu;
      output_col_cpu.CopyFrom(output_col, platform::CPUPlace(),
                              ctx.device_context());
      auto* output_col_ptr = output_col_cpu.data<T>();

      for (size_t j = 0; j < batch_size; ++j) {
        output_cpu_ptr[i + j * weight_dims[0]] = output_col_ptr[j];
      }
    }

    if (bias) {
      // Copy the bias to cpu.
      Tensor bias_cpu;
      bias_cpu.CopyFrom(*bias, platform::CPUPlace(), ctx.device_context());
      auto bias_vec = EigenMatrix<T>::From(bias_cpu);
      Eigen::DSizes<int, 2> bcast(batch_size, 1);
      output_cpu_mat.device(cpu_place) =
          bias_vec.broadcast(bcast) + output_cpu_mat;
    }

    // Copy the output to gpu.
    out->CopyFrom(output_cpu, platform::GPUPlace(), ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    bilinear_tensor_product,
    ops::BilinearTensorProductCUDAKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    bilinear_tensor_product_grad,
    ops::BilinearTensorProductGradKernel<paddle::platform::GPUPlace, float>);
