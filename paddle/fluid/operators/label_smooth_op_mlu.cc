// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class LabelSmoothMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<phi::DenseTensor>("X");
    auto* dist_t = ctx.Input<phi::DenseTensor>("PriorDist");
    auto* out_t = ctx.Output<phi::DenseTensor>("Out");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto epsilon_gt = 1.0f - epsilon;

    if (in_t->numel() == 0) return;
    out_t->mutable_data<T>(ctx.GetPlace());
    auto label_dim = in_t->dims()[in_t->dims().size() - 1];

    MLUCnnlTensorDesc x_desc(*in_t);
    MLUCnnlTensorDesc out_desc(*out_t);
    auto data_type = ToCnnlDataType<T>();
    MLUCnnlOpTensorDesc op_tensor_desc(
        CNNL_OP_TENSOR_ADD, data_type, CNNL_NOT_PROPAGATE_NAN);
    if (ctx.HasInput("PriorDist")) {
      MLUCnnlTensorDesc dist_desc(*dist_t);
      MLUCnnl::OpTensor(ctx,
                        op_tensor_desc.get(),
                        x_desc.get(),
                        GetBasePtr(in_t),
                        dist_desc.get(),
                        GetBasePtr(dist_t),
                        out_desc.get(),
                        GetBasePtr(out_t),
                        data_type,
                        epsilon_gt,
                        epsilon);
    } else {
      auto& dev_ctx = ctx.template device_context<MLUDeviceContext>();
      phi::DenseTensor dist_tensor =
          ctx.AllocateTmpTensor<T, MLUDeviceContext>({1, label_dim}, dev_ctx);
      MLUCnnlTensorDesc dist_desc(dist_tensor);
      auto value = static_cast<T>(1.0f / label_dim);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value,
                    dist_desc.get(),
                    GetBasePtr(&dist_tensor));
      MLUCnnl::OpTensor(ctx,
                        op_tensor_desc.get(),
                        x_desc.get(),
                        GetBasePtr(in_t),
                        dist_desc.get(),
                        GetBasePtr(&dist_tensor),
                        out_desc.get(),
                        GetBasePtr(out_t),
                        data_type,
                        epsilon_gt,
                        epsilon);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(label_smooth,
                       ops::LabelSmoothMLUKernel<float>,
                       ops::LabelSmoothMLUKernel<plat::float16>);
