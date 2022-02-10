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

#include <cstring>

#include "paddle/fluid/operators/mkldnn/axpy_handler.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace pplat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class SGDOneDNNKernel : public SGDOpKernel<pplat::CPUDeviceContext, T> {
 protected:
  void dense_param_and_grad_kernel(
      const framework::ExecutionContext &ctx) const override {
    VLOG(4) << "[ONEDNN]: sgd_dense_param_kernel<T, LodTensor>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    const auto *param = ctx.Input<framework::Tensor>("Param");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<framework::Tensor>("Grad");

    auto *out_data = param_out->mutable_data<T>(ctx.GetPlace());
    const T *param_data = param->data<T>();
    const auto *grad_data = grad->data<T>();
    const auto *lr = learning_rate->data<T>();
    // Since denese SGD is not in place operation, first copy params to output
    // tensor and then update it.
    std::memcpy(out_data, param_data, param->memory_size());
    OneDNNAXPYHandler<T>(param_out->numel(), -lr[0])(grad_data, out_data);
  }

  void dense_param_sparse_grad_kernel(
      const framework::ExecutionContext &ctx) const override {
    VLOG(4) << "[ONEDNN]: sgd_dense_param_kernel<T, SelectedRows>";
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
    const auto *grad = ctx.Input<pten::SelectedRows>("Grad");

    const auto &grad_value = grad->value();
    const auto &grad_rows = grad->rows();
    const auto grad_height = grad->height();
    const int64_t grad_val_height = static_cast<int64_t>(grad_rows.size());
    const auto grad_width = grad_value.numel() / grad_val_height;

    const auto *grad_data = grad_value.data<T>();
    auto *out_data = param_out->data<T>();
    const auto *lr = learning_rate->data<T>();

    OneDNNAXPYHandler<T> axpy_handler(grad_width, -lr[0]);

    for (size_t i = 0; i < grad_rows.size(); ++i) {
      PADDLE_ENFORCE_LT(
          grad_rows[i], grad_height,
          pplat::errors::OutOfRange(
              "Grad rows index value should be less than grad height."
              "Got [%s], but expected less than [%s]",
              grad_rows[i], grad_height));
      const int64_t row = grad_rows[i];
      const auto *src = grad_data + i * grad_width;
      auto *dst = out_data + row * grad_width;
      axpy_handler(src, dst);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(sgd, MKLDNN, pplat::CPUPlace, ops::SGDOneDNNKernel<float>,
                   ops::SGDOneDNNKernel<pplat::bfloat16>);
