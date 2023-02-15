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
class MeshgridMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    auto outs = ctx.MultiOutput<phi::DenseTensor>("Out");
    PADDLE_ENFORCE_EQ((ins.size() > 1) && (ins.size() < 7),
                      true,
                      platform::errors::InvalidArgument(
                          "Excepted phi::DenseTensor numbers between 2 and 6, "
                          "but only received d% .",
                          ins.size()));

    int64_t size = ins.size();
    std::vector<int64_t> shape(size);

    for (int64_t i = 0; i < size; i++) {
      switch (ins[i]->dims().size()) {
        case 0:
          shape[i] = 1;
          break;
        case 1:
          shape[i] = ins[i]->dims()[0];
          break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Expected scalar or 1D tensor in the tensor list but got tensor "
              "%d: ",
              i));
      }
    }

    MLUCnnlTensorDesc out_desc(size, shape.data(), ToCnnlDataType<T>());
    framework::DDim out_dims = phi::make_ddim(shape);
    for (int64_t i = 0; i < size; i++) {
      std::vector<int64_t> view_shape(size, 1);
      view_shape[i] = shape[i];

      outs[i]->Resize(out_dims);
      outs[i]->mutable_data<T>(ctx.GetPlace());

      MLUCnnlTensorDesc in_desc(size, view_shape.data(), ToCnnlDataType<T>());
      MLUCnnl::BroadcastTo(ctx,
                           in_desc.get(),
                           GetBasePtr(ins[i]),
                           out_desc.get(),
                           GetBasePtr(outs[i]));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_MLU_KERNEL(
    meshgrid,
    paddle::operators::MeshgridMLUKernel<int>,
    paddle::operators::MeshgridMLUKernel<float>,
    paddle::operators::MeshgridMLUKernel<int64_t>,
    paddle::operators::MeshgridMLUKernel<paddle::platform::float16>);
