/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AddPositionEncodingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::LoDTensor>("X");
    auto& x_lod = X->lod();
    auto* src_ptr = X->data<T>();

    auto* Out = context.Output<framework::LoDTensor>("Out");
    auto* dst_ptr = Out->mutable_data<T>(context.GetPlace());

    float alpha = context.Attr<float>("alpha");
    float beta = context.Attr<float>("beta");

    auto x_dim = X->dims();
    int batch_size = 0;
    int max_seq_len = 0;
    int enc_size = 0;

    if (x_lod.empty()) {
      PADDLE_ENFORCE(
          x_dim.size() == 3UL,
          "The input X of Add Position Encoding should be 3-D Tensor!");
      batch_size = x_dim[0];
      max_seq_len = x_dim[1];
      enc_size = x_dim[2];
    } else {
      PADDLE_ENFORCE(
          x_dim.size() == 2UL,
          "The input X of Add Position Encoding should be 2-D LoDTensor!");
      PADDLE_ENFORCE(
          x_lod.size() == 1UL,
          "The Add Position Encoding Op only supports lod_level == 1!");
      batch_size = x_lod[0].size() - 1;
      max_seq_len = -1;
      enc_size = x_dim[1];
    }

    PADDLE_ENFORCE(enc_size % 2 == 0, "Only support even encode size!");

    const int half_size = enc_size / 2;
    for (int i = 0; i < batch_size; ++i) {
      const int max_length =
          x_lod.empty() ? max_seq_len : x_lod[0][i + 1] - x_lod[0][i];
      for (int j = 0; j < max_length; ++j) {
        for (int k = 0; k < half_size; ++k) {
          const double val =
              (half_size > 1)
                  ? j / pow(10000.0, static_cast<double>(k) / (half_size - 1))
                  : j / 10000.0;
          dst_ptr[k] = src_ptr[k] * alpha + sin(val) * beta;
          dst_ptr[half_size + k] =
              src_ptr[half_size + k] * alpha + cos(val) * beta;
        }
        src_ptr += enc_size;
        dst_ptr += enc_size;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class AddPositionEncodingGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dOut =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto dout = framework::EigenVector<T>::Flatten(*dOut);

    auto* dX =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());
    auto dx = framework::EigenVector<T>::Flatten(*dX);

    float alpha = context.Attr<float>("alpha");

    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    dx.device(*place) = dout * static_cast<T>(alpha);
  }
};

}  // namespace operators
}  // namespace paddle
