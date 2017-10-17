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

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"

namespace paddle {
namespace operators {

inline void get_blockexpand_output_shape(int img_height, int img_width,
                                         int block_height, int block_width,
                                         int stride_height, int stride_width,
                                         int padding_height, int padding_width,
                                         int& outputHeight, int& outputWidth) {
  outputHeight =
      1 +
      (img_height + 2 * padding_height - block_height + stride_height - 1) /
          stride_height;

  outputWidth =
      1 +
      (img_width + 2 * padding_width - block_width + stride_width - 1) /
          stride_width;
}

template <typename Place, typename T>
class BlockExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using namespace framework;
    const Tensor* in = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto in_dim = in->dims();
    int N = in_dim[0];
    int C = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int block_height = ctx.Attr<int>("blockHeight");
    int block_width = ctx.Attr<int>("blockWidth");
    int stride_height = ctx.Attr<int>("strideHeight");
    int stride_width = ctx.Attr<int>("strideWidth");
    int padding_height = ctx.Attr<int>("paddingHeight");
    int padding_width = ctx.Attr<int>("paddingWidth");

    int outputHeight = 0;
    int outputWidth = 0;

    get_blockexpand_output_shape(
        img_height, img_width, block_height, block_width, stride_height,
        stride_width, padding_height, padding_width, outputHeight, outputWidth);

    for (int i = 0; i < N; i++) {
      Tensor src = in->Slice<T>(i, i + 1).Resize({C, img_height, img_width});
      Tensor dst = out->Slice<T>(i, i + 1).Resize(
          {outputHeight, outputWidth, C, block_height, block_width});
      math::Im2ColFunctor<math::ColFormat::kOCF, Place, T> f;
      f(ctx.device_context(), src, dst, stride_height, stride_width,
        padding_height, padding_width);
    }
  }
};

template <typename Place, typename T>
class BlockExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using namespace framework;
    auto* in = ctx.Input<Tensor>("X");
    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<Tensor>(GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto x_v = framework::EigenVector<T>::Flatten(*d_x);
    x_v.device(ctx.GetEigenDevice<Place>()) = x_v.constant(0.0);

    auto in_dim = in->dims();
    int N = in_dim[0];
    int C = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int block_height = ctx.Attr<int>("blockHeight");
    int block_width = ctx.Attr<int>("blockWidth");
    int stride_height = ctx.Attr<int>("strideHeight");
    int stride_width = ctx.Attr<int>("strideWidth");
    int padding_height = ctx.Attr<int>("paddingHeight");
    int padding_width = ctx.Attr<int>("paddingWidth");

    int outputHeight = 0;
    int outputWidth = 0;

    get_blockexpand_output_shape(
        img_height, img_width, block_height, block_width, stride_height,
        stride_width, padding_height, padding_width, outputHeight, outputWidth);

    for (int i = 0; i < N; i++) {
      Tensor dst = d_x->Slice<T>(i, i + 1).Resize({C, img_height, img_width});
      Tensor src = d_out->Slice<T>(i, i + 1).Resize(
          {outputHeight, outputWidth, C, block_height, block_width});
      math::Col2ImFunctor<math::ColFormat::kOCF, Place, T> f;
      f(ctx.device_context(), dst, src, stride_height, stride_width,
        padding_height, padding_width);
    }
  }
};

}  // namespace operators
}  // namespace paddle
