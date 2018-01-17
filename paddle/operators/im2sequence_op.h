/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/data_layout.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

inline int get_output_size(int img_size, int block_size, int stride,
                           int padding) {
  return (1 + (img_size + 2 * padding - block_size + stride - 1) / stride);
}

template <typename DeviceContext, typename T>
class Im2SequenceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* in = ctx.Input<Tensor>("X");
    LoDTensor* out = ctx.Output<LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    // TODO(wanghaoshuang): Add layout checker after 'set_layout'
    // being available for python API
    // PADDLE_ENFORCE_EQ(in->layout(), framework::DataLayout::kNCHW,
    //                  "Input(X) layout must be NCHW");
    auto in_dim = in->dims();
    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];
    int block_height = ctx.Attr<int>("block_height");
    int block_width = ctx.Attr<int>("block_width");
    int stride_height = ctx.Attr<int>("stride_height");
    int stride_width = ctx.Attr<int>("stride_width");
    int padding_height = ctx.Attr<int>("padding_height");
    int padding_width = ctx.Attr<int>("padding_width");

    int output_height = get_output_size(img_height, block_height, stride_height,
                                        padding_height);
    int output_width =
        get_output_size(img_width, block_width, stride_width, padding_width);

    const std::vector<int> dilations({1, 1});
    const std::vector<int> strides(
        {stride_height, stride_width, stride_height, stride_width});
    const std::vector<int> paddings(
        {padding_height, padding_width, padding_height, padding_width});

    auto out_dims = out->dims();
    out->Resize({batch_size, out->numel() / batch_size});
    for (int i = 0; i < batch_size; i++) {
      const Tensor src =
          in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
      Tensor dst = out->Slice(i, i + 1).Resize({output_height, output_width,
                                                img_channels, block_height,
                                                block_width});

      math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      f(dev_ctx, src, dilations, strides, paddings, &dst);
    }
    out->Resize(out_dims);

    // set lod information
    // TODO(wanghaoshuang): Move this to InferShape
    framework::LoD lod(1);
    lod[0].reserve(batch_size + 1);
    for (int i = 0, offset = 0; i < batch_size + 1; ++i) {
      lod[0][i] = offset;
      offset += output_height * output_width;
    }
    out->set_lod(lod);
  }
};

template <typename DeviceContext, typename T>
class Im2SequenceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<Tensor>("X");
    Tensor* d_out =
        const_cast<Tensor*>(ctx.Input<Tensor>(framework::GradVarName("Out")));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto x_v = framework::EigenVector<T>::Flatten(*d_x);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    x_v.device(place) = x_v.constant(0.0);

    auto in_dim = in->dims();
    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int block_height = ctx.Attr<int>("block_height");
    int block_width = ctx.Attr<int>("block_width");
    int stride_height = ctx.Attr<int>("stride_height");
    int stride_width = ctx.Attr<int>("stride_width");
    int padding_height = ctx.Attr<int>("padding_height");
    int padding_width = ctx.Attr<int>("padding_width");
    int output_height = get_output_size(img_height, block_height, stride_height,
                                        padding_height);
    int output_width =
        get_output_size(img_width, block_width, stride_width, padding_width);

    const std::vector<int> dilations({1, 1});
    const std::vector<int> strides(
        {stride_height, stride_width, stride_height, stride_width});
    const std::vector<int> paddings(
        {padding_height, padding_width, padding_height, padding_width});

    auto d_out_dims = d_out->dims();
    d_out->Resize({batch_size, d_out->numel() / batch_size});
    for (int i = 0; i < batch_size; i++) {
      Tensor dst =
          d_x->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
      const Tensor src = d_out->Slice(i, i + 1).Resize(
          {output_height, output_width, img_channels, block_height,
           block_width});
      math::Col2ImFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      f(dev_ctx, src, dilations, strides, paddings, &dst);
    }
    d_out->Resize(d_out_dims);
  }
};

}  // namespace operators
}  // namespace paddle
