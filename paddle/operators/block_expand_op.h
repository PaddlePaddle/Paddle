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

namespace paddle {
namespace operators {

template <typename Place, typename T>
class BlockExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using namespace framework;
    const Tensor* in = ctx.Input<Tensor>("input");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto in_dim = in->dims();
    int N = in_dim[0];
    int C = in_dim[1];

    int in_height = in_dim[2];
    int in_width = in_dim[3];

    int block_height = ctx.Attr<int>("block_height");
    int block_width = ctx.Attr<int>("block_width");
    int stride_height = ctx.Attr<int>("stride_height");
    int stride_width = ctx.Attr<int>("stride_width");
    int padding_height = ctx.Attr<int>("padding_height");
    int padding_width = ctx.Attr<int>("padding_width");

    int output_height =
        1 +
        (in_height + 2 * padding_height - block_height + stride_height - 1) /
            stride_height;

    int output_width =
        1 +
        (in_width + 2 * padding_width - block_width + stride_width - 1) /
            stride_width;

    Tensor col;
    if (clo_format = KCFO) {
      col.Resize(
          {N, C, filter_height, filter_width, output_height, output_width});
    } else {
      col.Resize(
          {N, output_height, output_width, C, filter_height, filter_width});
    }

    for (size_t i = 0; i < N; i++) {
      Im2ColFunctor<col_format, place, T>(ctx, one_img, col, stride[0],
                                          stride[1], padding[0], padding[1]);
    }
  }
};

template <typename Place, typename T>
class BlockExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    /*
  int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
  int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
  const Tensor* x = ctx.Input<Tensor>("X");
  const Tensor* y = ctx.Input<Tensor>("Y");
  */
  }
};

}  // namespace operators
}  // namespace paddle
