/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/pooling.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>
class SppKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    int pyramid_height = context.template Attr<int>("pyramid_height");
    out->mutable_data<T>(context.GetPlace());
    auto out_stride = framework::stride(out->dims());
    int input_h = in_x->dims()[2];
    int input_w = in_x->dims()[3];
    size_t output_offset = 0;
    for (int p = 0; p < pyramid_height; ++p) {
      int bins = std::pow(2, p);
      int ksize_h = std::ceil(input_h / static_cast<double>(bins));
      int ksize_w = std::ceil(input_w / static_cast<double>(bins));
      int padding_h = (ksize_h * bins - input_h + 1) / 2;
      int padding_w = (ksize_w * bins - input_w + 1) / 2;
      std::vector<int> ksize({ksize_h, ksize_w});
      std::vector<int> strides({ksize_h, ksize_w});
      std::vector<int> paddings({padding_h, padding_w});
      // pooling output shape
      framework::Tensor out_level;
      std::vector<int64_t> output_shape_vec({in_x->dims()[0], in_x->dims()[1]});
      output_shape_vec.push_back((input_h - ksize_h + 2 * padding_h) / ksize_h +
                                 1);
      output_shape_vec.push_back((input_w - ksize_w + 2 * padding_w) / ksize_w +
                                 1);
      framework::DDim output_shape(framework::make_ddim(output_shape_vec));
      out_level.mutable_data<T>(output_shape, context.GetPlace());
      // pooling
      math::Pool2dFunctor<Place, math::MaxPool<T>, T> pool_forward;
      math::MaxPool<T> max_process;
      pool_forward(context.device_context(), *in_x, ksize, strides, paddings,
                   max_process, &out_level);
      // flatten pooling output shape
      framework::Tensor out_flatten_level;
      int output_flatten_w = in_x->dims()[1] * bins * bins;
      std::vector<int64_t> output_flatten_shape_vec(
          {in_x->dims()[0], output_flatten_w});
      framework::DDim output_flatten_shape(
          framework::make_ddim(output_flatten_shape_vec));
      out_flatten_level.ShareDataWith(out_level);
      out_flatten_level.Resize(output_flatten_shape);
      // concat
      auto out_flatten_level_stride =
          framework::stride(out_flatten_level.dims());
      StridedMemcpy<T>(context.device_context(), out_flatten_level.data<T>(),
                       out_flatten_level_stride, out_flatten_level.dims(),
                       out_stride, out->data<T>() + output_offset);
      output_offset +=
          out_flatten_level.dims()[1] * out_flatten_level_stride[1];
    }
  }
};
template <typename Place, typename T>
class SppGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    const framework::Tensor* out = context.Input<framework::Tensor>("Out");
    const framework::Tensor* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* in_x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    int pyramid_height = context.template Attr<int>("pyramid_height");
    auto& device_ctx = context.device_context();
    math::SetConstant<Place, T> zero;
    in_x_grad->mutable_data<T>(context.GetPlace());
    zero(device_ctx, in_x_grad, static_cast<T>(0));
    auto out_stride = framework::stride(out->dims());
    int input_h = in_x->dims()[2];
    int input_w = in_x->dims()[3];
    size_t out_offset = 0;
    for (int p = 0; p < pyramid_height; ++p) {
      int bins = std::pow(2, p);
      int ksize_h = std::ceil(input_h / static_cast<double>(bins));
      int ksize_w = std::ceil(input_w / static_cast<double>(bins));
      int padding_h = (ksize_h * bins - input_h + 1) / 2;
      int padding_w = (ksize_w * bins - input_w + 1) / 2;
      std::vector<int> ksize({ksize_h, ksize_w});
      std::vector<int> strides({ksize_h, ksize_w});
      std::vector<int> paddings({padding_h, padding_w});
      // split out and outgrad  ...  to flatten
      framework::Tensor out_flatten_level;
      framework::Tensor outgrad_flatten_level;
      int out_flatten_w = in_x->dims()[1] * bins * bins;
      std::vector<int64_t> out_flatten_shape_vec(
          {in_x->dims()[0], out_flatten_w});
      framework::DDim out_flatten_shape(
          framework::make_ddim(out_flatten_shape_vec));
      out_flatten_level.mutable_data<T>(out_flatten_shape, context.GetPlace());
      outgrad_flatten_level.mutable_data<T>(out_flatten_shape,
                                            context.GetPlace());
      auto flatten_stride = framework::stride(out_flatten_level.dims());
      // memcpy
      StridedMemcpy<T>(context.device_context(), out->data<T>() + out_offset,
                       out_stride, out_flatten_level.dims(), flatten_stride,
                       out_flatten_level.data<T>());

      StridedMemcpy<T>(context.device_context(),
                       out_grad->data<T>() + out_offset, out_stride,
                       outgrad_flatten_level.dims(), flatten_stride,
                       outgrad_flatten_level.data<T>());
      out_offset += out_flatten_level.dims()[1] * out_stride[1];
      // flatten backward to nchw
      framework::Tensor out_level;
      framework::Tensor outgrad_level;
      std::vector<int64_t> out_shape_vec({in_x->dims()[0], in_x->dims()[1]});
      out_shape_vec.push_back((input_h - ksize_h + 2 * padding_h) / ksize_h +
                              1);
      out_shape_vec.push_back((input_w - ksize_w + 2 * padding_w) / ksize_w +
                              1);
      framework::DDim out_shape(framework::make_ddim(out_shape_vec));
      out_level.ShareDataWith(out_flatten_level);
      out_level.Resize(out_shape);
      outgrad_level.ShareDataWith(outgrad_flatten_level);
      outgrad_level.Resize(out_shape);
      // pooling backward
      math::MaxPool2dGradFunctor<Place, T> pool2d_backward;
      pool2d_backward(context.device_context(), *in_x, *&out_level,
                      *&outgrad_level, ksize, strides, paddings, in_x_grad);
    }
  }
};
}  // namespace operators
}  // namespace paddle
