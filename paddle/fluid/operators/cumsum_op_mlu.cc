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
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CumSumMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool reverse = ctx.Attr<bool>("reverse");
    bool flatten = ctx.Attr<bool>("flatten");

    out->mutable_data<T>(ctx.GetPlace());

    Tensor* input_ptr = const_cast<Tensor*>(x);
    Tensor flat_x(x->type());
    if (flatten) {
      PADDLE_ENFORCE_EQ(
          axis, -1,
          platform::errors::InvalidArgument(
              "when flatten is true, attr axis must be default %d, but got %d",
              -1, axis));

      flat_x.ShareDataWith(*x);
      flat_x.Resize(phi::make_ddim({x->numel()}));
      input_ptr = &flat_x;
    }

    const int true_axis = (axis < 0) ? input_ptr->dims().size() + axis : axis;
    MLUCnnlTensorDesc input_desc(*input_ptr);
    MLUCnnlTensorDesc out_desc(*out);

    MLUCnnl::Cumsum(ctx, true_axis, exclusive, reverse, input_desc.get(),
                    GetBasePtr(input_ptr), out_desc.get(), GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(cumsum, ops::CumSumMLUKernel<int>,
                       ops::CumSumMLUKernel<float>,
                       ops::CumSumMLUKernel<plat::float16>);
