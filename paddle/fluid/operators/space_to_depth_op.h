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
#ifndef PADDLE_FLUID_OPERATORS_SPACE_TO_DEPTH_OP_H_
#define PADDLE_FLUID_OPERATORS_SPACE_TO_DEPTH_OP_H_
#endif  // PADDLE_FLUID_OPERATORS_SPACE_TO_DEPTH_OP_H_

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
class space_to_depth_compute {
 public:
  HOSTDEVICE space_to_depth_compute(const T *x, int64_t w, int64_t h, int64_t c,
                                    int64_t batch, int64_t blocksize,
                                    int64_t forward, T *out)
      : x_(x),
        w_(w),
        h_(h),
        c_(c),
        batch_(batch),
        blocksize_(blocksize),
        forward_(forward),
        out_(out) {}

  HOSTDEVICE void operator()(int64_t in_index) {
    int64_t out_c = c_ / (blocksize_ * blocksize_);
    // calculate each dim position with index of tensor
    int64_t b = in_index / (c_ * h_ * w_);
    int64_t k = (in_index % (c_ * h_ * w_)) / (h_ * w_);
    int64_t j = ((in_index % (c_ * h_ * w_)) % (h_ * w_)) / w_;
    int64_t i = ((in_index % (c_ * h_ * w_)) % (h_ * w_)) % w_;

    int64_t c2 = k % out_c;
    int64_t offset = k / out_c;
    int64_t w2 = i * blocksize_ + offset % blocksize_;
    int64_t h2 = j * blocksize_ + offset / blocksize_;
    int64_t out_index =
        w2 + w_ * blocksize_ * (h2 + h_ * blocksize_ * (c2 + out_c * b));
    if (forward_)
      out_[out_index] = x_[in_index];
    else
      out_[in_index] = x_[out_index];
  }

 private:
  const T *x_;
  int64_t w_, h_, c_, batch_, blocksize_, forward_;
  T *out_;
};

template <typename DeviceContext, typename T>
class SpaceToDepthKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<framework::LoDTensor>("Out");
    auto *x = context.Input<framework::LoDTensor>("X");
    auto blocksize = context.Attr<int64_t>("blocksize");
    auto in_dims = x->dims();
    out->mutable_data(context.GetPlace(), x->type());

    auto out_dims = out->dims();
    auto B = in_dims[0];
    auto C = in_dims[1];
    auto H = in_dims[2];
    auto W = in_dims[3];
    platform::ForRange<DeviceContext> for_range(
        context.template device_context<DeviceContext>(),
        static_cast<size_t>(x->numel()));

    auto *x_data = x->data<T>();
    auto *out_data = out->data<T>();
    paddle::operators::space_to_depth_compute<T> computer(
        x_data, W, H, C, B, blocksize, 1, out_data);
    for_range(computer);

    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SpaceToDepthGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *d_out =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *d_x =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto blocksize = context.Attr<int64_t>("blocksize");
    auto in_dims = d_x->dims();
    d_x->mutable_data(context.GetPlace(), d_out->type());

    auto B = in_dims[0];
    auto C = in_dims[1];
    auto H = in_dims[2];
    auto W = in_dims[3];

    platform::ForRange<DeviceContext> for_range(
        context.template device_context<DeviceContext>(),
        static_cast<size_t>(d_x->numel()));

    auto *dx_data = d_x->data<T>();
    auto *dout_data = d_out->data<T>();

    paddle::operators::space_to_depth_compute<T> computer(
        dout_data, W, H, C, B, blocksize, 0, dx_data);
    for_range(computer);

    d_x->Resize(in_dims);
  }
};

}  // namespace operators
}  // namespace paddle
