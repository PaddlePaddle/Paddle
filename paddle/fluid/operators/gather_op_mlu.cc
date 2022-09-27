/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class GatherOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto axis = ctx.Attr<int>("axis");

    const auto index_dims = index->dims();
    if (index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          index_dims[1],
          1,
          platform::errors::InvalidArgument(
              "The last dim of index should be 1 when it is 2D, but we get %d",
              index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          index_dims.size(),
          1,
          platform::errors::InvalidArgument(
              "The index should be 1D, when it is not 2D, but we get %d",
              index_dims.size()));
    }

    auto *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    int index_shape_1d[1] = {static_cast<int>(index_dims[0])};
    MLUCnnlTensorDesc index_desc(
        1, index_shape_1d, ToCnnlDataType(index->dtype()));
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::GatherFunctor(ctx,
                           axis,
                           0 /*batch_dims*/,
                           x_desc.get(),
                           GetBasePtr(x),
                           index_desc.get(),
                           GetBasePtr(index),
                           out_desc.get(),
                           GetBasePtr(out));
  }
};

template <typename T>
class GatherGradOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *index = ctx.Input<Tensor>("Index");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    const auto index_dims = index->dims();
    if (index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          index_dims[1],
          1,
          platform::errors::InvalidArgument(
              "The last dim of index should be 1 when it is 2D, but we get %d",
              index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          index_dims.size(),
          1,
          platform::errors::InvalidArgument(
              "The index should be 1D, when it is not 2D, but we get %d",
              index_dims.size()));
    }

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc dx_desc(*dx);
    auto value = static_cast<T>(0);
    MLUCnnl::Fill(
        ctx, CNNL_POINTER_MODE_HOST, &value, dx_desc.get(), GetBasePtr(dx));

    int index_shape_1d[1] = {static_cast<int>(index_dims[0])};
    MLUCnnlTensorDesc index_desc(
        1, index_shape_1d, ToCnnlDataType(index->dtype()));
    MLUCnnlTensorDesc dout_desc(*dout);
    const cnnlScatterRefMode_t mode = CNNL_SCATTERREF_UPDATE;
    MLUCnnl::ScatterRefFunctor(ctx,
                               dx_desc.get(),
                               GetBasePtr(dx),
                               dout_desc.get(),
                               GetBasePtr(dout),
                               index_desc.get(),
                               GetBasePtr(index),
                               mode);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(gather,
                       ops::GatherOpMLUKernel<float>,
                       ops::GatherOpMLUKernel<paddle::platform::float16>,
                       ops::GatherOpMLUKernel<int>);

REGISTER_OP_MLU_KERNEL(gather_grad,
                       ops::GatherGradOpMLUKernel<float>,
                       ops::GatherGradOpMLUKernel<paddle::platform::float16>,
                       ops::GatherGradOpMLUKernel<int>);
