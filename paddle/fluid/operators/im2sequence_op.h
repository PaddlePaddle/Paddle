/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

inline int Im2SeqOutputSize(int input_size, int filter_size, int padding_0,
                            int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

template <typename DeviceContext, typename T>
class Im2SequenceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* in = ctx.Input<Tensor>("X");
    LoDTensor* out = ctx.Output<LoDTensor>("Out");
    auto in_dim = in->dims();
    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];
    auto kernels = ctx.Attr<std::vector<int>>("kernels");
    auto strides = ctx.Attr<std::vector<int>>("strides");
    auto paddings = ctx.Attr<std::vector<int>>("paddings");
    if (ctx.HasInput("Y") && batch_size > 1) {
      const Tensor* imgrealsize = ctx.Input<Tensor>("Y");
      auto out_stride = ctx.Attr<std::vector<int>>("out_stride");
      Tensor cpu_shape_tensor;
      paddle::framework::TensorCopySync(*imgrealsize, platform::CPUPlace(),
                                        &cpu_shape_tensor);
      std::vector<int> imgreal_h;
      std::vector<int> imgreal_w;
      std::vector<int> output_height;
      std::vector<int> output_width;
      int result = 0;
      for (int i = 0; i < batch_size; i++) {
        int tmp_real_h = static_cast<int>((cpu_shape_tensor.data<T>())[2 * i]);
        int tmp_real_w =
            static_cast<int>((cpu_shape_tensor.data<T>())[2 * i + 1]);
        if (tmp_real_h % out_stride[0] == 0) {
          tmp_real_h = tmp_real_h / out_stride[0];
        } else {
          tmp_real_h = tmp_real_h / out_stride[0] + 1;
        }
        if (tmp_real_w % out_stride[1] == 0) {
          tmp_real_w = tmp_real_w / out_stride[1];
        } else {
          tmp_real_w = tmp_real_w / out_stride[1] + 1;
        }
        imgreal_h.push_back(tmp_real_h);
        imgreal_w.push_back(tmp_real_w);
        output_height.push_back(Im2SeqOutputSize(
            imgreal_h[i], kernels[0], paddings[0], paddings[2], strides[0]));
        output_width.push_back(Im2SeqOutputSize(
            imgreal_w[i], kernels[1], paddings[1], paddings[3], strides[1]));
        result += output_height[i] * output_width[i];
      }

      out->mutable_data<T>({result, img_channels * kernels[0] * kernels[1]},
                           ctx.GetPlace());

      const std::vector<int> dilations({1, 1});
      int offset_out = 0;
      for (int i = 0; i < batch_size; i++) {
        const Tensor src =
            in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
        Tensor dst = out->Slice(offset_out,
                                offset_out + output_height[i] * output_width[i])
                         .Resize({output_height[i], output_width[i],
                                  img_channels, kernels[0], kernels[1]});
        offset_out += output_height[i] * output_width[i];

        math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        f(dev_ctx, src, dilations, strides, paddings, &dst);
      }
      framework::LoD lod(1);
      lod[0].reserve(batch_size + 1);
      int offset = 0;
      lod[0].push_back(offset);
      for (int i = 0; i < batch_size; ++i) {
        offset += output_height[i] * output_width[i];
        lod[0].push_back(offset);
      }
      out->set_lod(lod);
    } else {
      int output_height = Im2SeqOutputSize(img_height, kernels[0], paddings[0],
                                           paddings[2], strides[0]);
      int output_width = Im2SeqOutputSize(img_width, kernels[1], paddings[1],
                                          paddings[3], strides[1]);
      out->mutable_data<T>(
          {static_cast<int64_t>(batch_size) * output_height * output_width,
           static_cast<int64_t>(img_channels) * kernels[0] * kernels[1]},
          ctx.GetPlace());
      const std::vector<int> dilations({1, 1});
      auto out_dims = out->dims();
      out->Resize({batch_size, out->numel() / batch_size});
      for (int i = 0; i < batch_size; i++) {
        const Tensor src =
            in->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
        Tensor dst =
            out->Slice(i, i + 1).Resize({output_height, output_width,
                                         img_channels, kernels[0], kernels[1]});

        math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        f(dev_ctx, src, dilations, strides, paddings, &dst);
      }
      out->Resize(out_dims);
      framework::LoD lod(1);
      lod[0].reserve(batch_size + 1);
      int offset = 0;
      lod[0].push_back(offset);
      for (int i = 0; i < batch_size; ++i) {
        offset += output_height * output_width;
        lod[0].push_back(offset);
      }
      out->set_lod(lod);
    }
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
    EigenConstant<std::decay_t<decltype(place)>, T, 1>::Eval(place, x_v, 0.0);

    auto in_dim = in->dims();
    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    auto kernels = ctx.Attr<std::vector<int>>("kernels");
    auto strides = ctx.Attr<std::vector<int>>("strides");
    auto paddings = ctx.Attr<std::vector<int>>("paddings");
    int output_height = Im2SeqOutputSize(img_height, kernels[0], paddings[0],
                                         paddings[2], strides[0]);
    int output_width = Im2SeqOutputSize(img_width, kernels[1], paddings[1],
                                        paddings[3], strides[1]);

    const std::vector<int> dilations({1, 1});

    auto d_out_dims = d_out->dims();
    d_out->Resize({batch_size, d_out->numel() / batch_size});
    for (int i = 0; i < batch_size; i++) {
      Tensor dst =
          d_x->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
      const Tensor src = d_out->Slice(i, i + 1).Resize(
          {output_height, output_width, img_channels, kernels[0], kernels[1]});
      math::Col2ImFunctor<math::ColFormat::kOCF, DeviceContext, T> f;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      f(dev_ctx, src, dilations, strides, paddings, &dst);
    }
    d_out->Resize(d_out_dims);
  }
};

}  // namespace operators
}  // namespace paddle
