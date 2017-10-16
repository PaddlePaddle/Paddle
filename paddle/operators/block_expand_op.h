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
#include "paddle/operators/math/img2col.h"

namespace paddle {
namespace operators {

inline void get_blockexpand_output_shape(int imgHeight, int imgWidth,
                                         int blockHeight, int blockWidth,
                                         int strideHeight, int strideWidth,
                                         int paddingHeight, int paddingWidth,
                                         int& outputHeight, int& outputWidth) {
  outputHeight =
      1 +
      (imgHeight + 2 * paddingHeight - blockHeight + strideHeight - 1) /
          strideHeight;

  outputWidth = 1 +
                (imgWidth + 2 * paddingWidth - blockWidth + strideWidth - 1) /
                    strideWidth;
}

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
    int imgHeight = in_dim[2];
    int imgWidth = in_dim[3];

    int blockHeight = ctx.Attr<int>("blockHeight");
    int blockWidth = ctx.Attr<int>("blockWidth");
    int strideHeight = ctx.Attr<int>("strideHeight");
    int strideWidth = ctx.Attr<int>("strideWidth");
    int paddingHeight = ctx.Attr<int>("paddingHeight");
    int paddingWidth = ctx.Attr<int>("paddingWidth");

    int outputHeight = 0;
    int outputWidth = 0;

    get_blockexpand_output_shape(imgHeight, imgWidth, blockHeight, blockWidth,
                                 strideHeight, strideWidth, paddingHeight,
                                 paddingWidth, outputHeight, outputWidth);

    for (int i = 0; i < N; i++) {
      Tensor src = in->Slice<T>(i, i + 1).Resize(C, imgHeight, imgWidth);
      Tensor dst = out->Slice<T>(i, i + 1).Resize(outputHeight, outputWidth, C,
                                                  blockHeight, blockWidth);
      math::Im2ColFunctor<kOCF, ctx->GetPlace(), T>(ctx, src, dst, strideHeight,
                                                    strideWidth, paddingHeight,
                                                    paddingWidth);
    }
  }
};

template <typename Place, typename T>
class BlockExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using namespace framework;
  }
};

}  // namespace operators
}  // namespace paddle
