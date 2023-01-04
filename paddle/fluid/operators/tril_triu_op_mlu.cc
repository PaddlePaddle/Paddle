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
template <typename T>
class TrilTriuMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    int diagonal = ctx.Attr<int>("diagonal");
    bool lower = ctx.Attr<bool>("lower");
    bool upper;
    if (lower) {
      upper = 0;
    } else {
      upper = 1;
    }

    out->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::TrilTriu(ctx,
                      diagonal,
                      upper,
                      x_desc.get(),
                      GetBasePtr(x),
                      out_desc.get(),
                      GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(tril_triu,
                       ops::TrilTriuMLUKernel<float>,
                       ops::TrilTriuMLUKernel<int32_t>,
                       ops::TrilTriuMLUKernel<plat::float16>);
