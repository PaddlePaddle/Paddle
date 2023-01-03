/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
class GatherNdNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *index = ctx.Input<phi::DenseTensor>("Index");
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    out->template mutable_data<T>(ctx.GetPlace());

    if (x->numel() == 0) return;

    if (index->numel() == 0) {
      framework::TensorCopy(*x, ctx.GetPlace(), ctx.device_context(), out);
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

    const auto &runner = NpuOpRunner("GatherNd", {*x, *index}, {*out}, {});
    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    runner.Run(stream);
  }
};

template <typename T>
class GatherNdGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *index = ctx.Input<phi::DenseTensor>("Index");
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *p = dx->mutable_data<T>(ctx.GetPlace());

    if (dx->numel() == 0) return;

    if (index->numel() == 0) {
      framework::TensorCopy(*dout, ctx.GetPlace(), ctx.device_context(), dx);
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

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    platform::NPUMemsetAsync(
        static_cast<void *>(p), 0, dx->numel() * sizeof(T), stream);

    const auto &runner_scatter = NpuOpRunner(
        "ScatterNdAdd", {*dx, *index, *dout}, {*dx}, {{"use_locking", false}});
    runner_scatter.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(gather_nd,
                       ops::GatherNdNPUKernel<paddle::platform::float16>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::GatherNdNPUKernel<int64_t>,
#endif
                       ops::GatherNdNPUKernel<float>);

REGISTER_OP_NPU_KERNEL(gather_nd_grad,
                       ops::GatherNdGradNPUKernel<paddle::platform::float16>,
                       ops::GatherNdGradNPUKernel<float>);
