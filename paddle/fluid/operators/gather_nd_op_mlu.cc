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
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class GatherNdMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *index = ctx.Input<phi::DenseTensor>("Index");
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    out->template mutable_data<T>(place);

    if (x->numel() == 0) return;
    if (index->numel() == 0) {
      auto &dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
      framework::TensorCopy(*x, place, dev_ctx, out);
      return;
    }

    const auto &index_type = framework::TransToProtoVarType(index->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match,
                      true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s],"
                          "but desires to be [%s] or [%s]",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc index_desc(*index);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::GatherNd(ctx,
                      x_desc.get(),
                      GetBasePtr(x),
                      index_desc.get(),
                      GetBasePtr(index),
                      out_desc.get(),
                      GetBasePtr(out));
  }
};

template <typename T>
class GatherNdGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *index = ctx.Input<phi::DenseTensor>("Index");
    auto *dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *x = ctx.Input<phi::DenseTensor>("X");

    if (dx->numel() == 0) return;
    if (index->numel() == 0) {
      auto &dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
      framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dx);
      return;
    }

    phi::DenseTensor tmp_tensor(index->type());
    phi::DenseTensor tmp_tensor2(dout->type());
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      tmp_tensor.ShareDataWith(*index);
      std::vector<int64_t> new_dim = {1, index_dims[0]};
      tmp_tensor.Resize(phi::make_ddim(new_dim));
      index = &tmp_tensor;

      tmp_tensor2.ShareDataWith(*dout);
      std::vector<int64_t> new_dim2{1};
      for (int i = index->numel(); i < x->dims().size(); i++) {
        new_dim2.push_back(x->dims()[i]);
      }
      tmp_tensor2.Resize(phi::make_ddim(new_dim2));
      dout = &tmp_tensor2;
    }

    dx->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc dx_desc(*dx);
    auto value = static_cast<T>(0);
    MLUCnnl::Fill(
        ctx, CNNL_POINTER_MODE_HOST, &value, dx_desc.get(), GetBasePtr(dx));

    MLUCnnlTensorDesc index_desc(*index);
    MLUCnnlTensorDesc dout_desc(*dout);

    const cnnlScatterNdMode_t mode = CNNL_SCATTERND_ADD;
    MLUCnnl::ScatterNd(ctx,
                       mode,
                       index_desc.get(),
                       GetBasePtr(index),
                       dout_desc.get(),
                       GetBasePtr(dout),
                       dx_desc.get(),
                       GetBasePtr(dx),
                       dx_desc.get(),
                       GetBasePtr(dx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(gather_nd,
                       ops::GatherNdMLUKernel<float>,
                       ops::GatherNdMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(gather_nd_grad,
                       ops::GatherNdGradMLUKernel<paddle::platform::float16>,
                       ops::GatherNdGradMLUKernel<float>);
