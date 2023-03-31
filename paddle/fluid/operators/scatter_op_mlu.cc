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
class ScatterMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* indices = ctx.Input<phi::DenseTensor>("Ids");
    auto* updates = ctx.Input<phi::DenseTensor>("Updates");
    bool overwrite = ctx.Attr<bool>("overwrite");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc indices_desc(*indices);
    MLUCnnlTensorDesc updates_desc(*updates);
    MLUCnnlTensorDesc out_desc(*out);
    cnnlScatterRefMode_t mode;
    if (overwrite) {
      mode = CNNL_SCATTERREF_UPDATE;
      MLUCnnl::ScatterRefFunctor(ctx,
                                 x_desc.get(),
                                 GetBasePtr(x),
                                 updates_desc.get(),
                                 GetBasePtr(updates),
                                 indices_desc.get(),
                                 GetBasePtr(indices),
                                 mode);
    } else {
      phi::DenseTensor tensor_zeros(updates->type());
      tensor_zeros.mutable_data<T>(updates->dims(), ctx.GetPlace());
      MLUCnnlTensorDesc tensor_zeros_desc(tensor_zeros);
      float value = 0.0;
      auto value_t = static_cast<T>(value);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    tensor_zeros_desc.get(),
                    GetBasePtr(&tensor_zeros));
      mode = CNNL_SCATTERREF_UPDATE;
      MLUCnnl::ScatterRefFunctor(ctx,
                                 x_desc.get(),
                                 GetBasePtr(x),
                                 tensor_zeros_desc.get(),
                                 GetBasePtr(&tensor_zeros),
                                 indices_desc.get(),
                                 GetBasePtr(indices),
                                 mode);
      mode = CNNL_SCATTERREF_ADD;
      MLUCnnl::ScatterRefFunctor(ctx,
                                 x_desc.get(),
                                 GetBasePtr(x),
                                 updates_desc.get(),
                                 GetBasePtr(updates),
                                 indices_desc.get(),
                                 GetBasePtr(indices),
                                 mode);
    }
    paddle::framework::TensorCopy(*x, place, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(scatter,
                       ops::ScatterMLUKernel<float>,
                       ops::ScatterMLUKernel<paddle::platform::float16>);
