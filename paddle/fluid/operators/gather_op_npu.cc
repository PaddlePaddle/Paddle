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

#include "paddle/fluid/operators/gather_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/kron_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GatherOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());
    auto runner = NpuOpRunner("Gather", {*x, *index}, {*out},
                              {{"validate_indices", true}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GatherGradOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *index = ctx.Input<Tensor>("Index");
    auto *x = ctx.Input<Tensor>("X");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    // step1: Unsqueeze index
    const auto index_dims = index->dims();
    if (index_dims.size() == 1) {
      framework::Tensor res;
      res.ShareDataWith(*index);
      std::vector<int64_t> new_dim = {index_dims[0], 1};
      res.Resize(framework::make_ddim(new_dim));
      index = &res;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // step2: ZerosLike x in device
    Tensor *tmp_zerox = const_cast<Tensor *>(x);
    Tensor zeroslike_xout(x->type());
    zeroslike_xout.Resize(x->dims());
    zeroslike_xout.mutable_data<T>(ctx.GetPlace());

    auto runner_zeroslike =
        NpuOpRunner("ZerosLike", {*x}, {zeroslike_xout}, {});
    runner_zeroslike.Run(stream);
    tmp_zerox = &zeroslike_xout;

    // step3: scatter(x_grad)
    dx->mutable_data<T>(ctx.GetPlace());
    auto runner_scatter = NpuOpRunner("TensorScatterUpdate",
                                      {*tmp_zerox, *index, *dout}, {*dx}, {});
    runner_scatter.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    gather, ops::GatherOpNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GatherOpNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    gather_grad,
    ops::GatherGradOpNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GatherGradOpNPUKernel<paddle::platform::NPUDeviceContext,
                               paddle::platform::float16>);
