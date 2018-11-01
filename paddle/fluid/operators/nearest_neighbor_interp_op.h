/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

template <typename T>
class NearestNeighborInterpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      auto out_size_data = out_size->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
    auto& device_ctx =
        ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, output, static_cast<T>(0.0));

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    float ratio_h =
        (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
    float ratio_w =
        (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

    auto input_t = EigenTensor<T, 4>::From(*input);
    auto output_t = EigenTensor<T, 4>::From(*output);
    for (int k = 0; k < out_h; k++) {  // loop for images
      int in_k = static_cast<int>(round(ratio_h * k));
      for (int l = 0; l < out_w; l++) {
        int in_l = static_cast<int>(round(ratio_w * l));
        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            output_t(i, j, k, l) = input_t(i, j, in_k, in_l);
          }
        }
      }
    }
  }
};

template <typename T>
class NearestNeighborInterpGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      auto out_size_data = out_size->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    input_grad->mutable_data<T>({n, c, in_h, in_w}, ctx.GetPlace());
    auto& device_ctx =
        ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, input_grad, static_cast<T>(0.0));

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    float ratio_h =
        (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
    float ratio_w =
        (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

    auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
    auto output_grad_t = EigenTensor<T, 4>::From(*output_grad);
    for (int k = 0; k < out_h; k++) {  // loop for images
      int in_k = static_cast<int>(round(ratio_h * k));
      for (int l = 0; l < out_w; l++) {
        int in_l = static_cast<int>(round(ratio_w * l));
        for (int i = 0; i < n; i++) {    // loop for batches
          for (int j = 0; j < c; j++) {  // loop for channels
            input_grad_t(i, j, in_k, in_l) += output_grad_t(i, j, k, l);
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
