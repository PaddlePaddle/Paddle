/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DropoutNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* seed_tensor =
        ctx.HasInput("Seed") ? ctx.Input<Tensor>("Seed") : nullptr;
    auto* out = ctx.Output<Tensor>("Out");
    auto* mask = ctx.Output<Tensor>("Mask");

    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    auto is_test = ctx.Attr<bool>("is_test");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    uint32_t mask_length = (x->numel() + 128 - 1) / 128 * 128;

    if (dropout_prob == 1.) {
      const auto& runner_zeros_out = NpuOpRunner("ZerosLike", {*out}, {*out});
      runner_zeros_out.Run(stream);
      mask->Resize(framework::make_ddim({mask_length / 8}));
      mask->mutable_data<uint8_t>(ctx.GetPlace());
      const auto& runner_zeros_mask =
          NpuOpRunner("ZerosLike", {*mask}, {*mask});
      runner_zeros_mask.Run(stream);
      return;
    }

    // only achive the default `upscale_in_train` method
    if (!is_test) {
      Tensor tmp_x(x->type());
      Tensor tmp_out(out->type());
      tmp_x.ShareDataWith(*x);
      tmp_out.ShareDataWith(*out);
      if (x->dims().size() == 1) {
        // dropout will get error result when input
        // is 1-D. Make it become 2-D.
        std::vector<int> vec_dim = framework::vectorize<int>(x->dims());
        tmp_x.Resize(framework::make_ddim({vec_dim[0], 1}));
        tmp_out.Resize(framework::make_ddim({vec_dim[0], 1}));
      }

      int seed = 0;
      int seed2 = 0;
      float keep_prob = 1. - dropout_prob;
      if (seed_tensor) {
        std::vector<int> seed_data;
        TensorToVector(*seed_tensor, ctx.device_context(), &seed_data);
        seed = seed_data[0];
      } else {
        seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : 0;
      }

      Tensor keep_prob_tensor(x->type());
      keep_prob_tensor.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&keep_prob_tensor,
                                   static_cast<T>(keep_prob));

      mask->Resize(framework::make_ddim({mask_length / 8}));
      mask->mutable_data<uint8_t>(ctx.GetPlace());

      // TODO(pangyoki): `keep_prob` used in `DropOutGenMask` NPU
      // OP must be a scalar with shape[0]. At present, the shape
      // of the `prob` Tensor of this OP is forced to be set to 0
      // in `npu_op_runner.cc`, which needs to be optimized later.
      NpuOpRunner runner_gen_mask;
      runner_gen_mask.SetType("DropOutGenMask")
          .AddInput(framework::vectorize(tmp_out.dims()))
          .AddInput(keep_prob_tensor)
          .AddOutput(*mask)
          .AddAttr("seed", seed)
          .AddAttr("seed2", seed2);
      runner_gen_mask.Run(stream);

      const auto& runner_dropout = NpuOpRunner(
          "DropOutDoMask", {tmp_x, *mask, keep_prob_tensor}, {tmp_out}, {});
      runner_dropout.Run(stream);
    } else {
      framework::TensorCopy(
          *x, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), out);
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = ctx.Input<Tensor>("Mask");

    auto dropout_prob = ctx.Attr<float>("dropout_prob");
    auto is_test = ctx.Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(is_test, false,
                      platform::errors::PreconditionNotMet(
                          "GradOp is only callable when is_test is false"));

    dx->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dropout_prob == 1.) {
      const auto& runner_zeros = NpuOpRunner("ZerosLike", {*dx}, {*dx});
      runner_zeros.Run(stream);
      return;
    }

    float keep_prob = 1. - dropout_prob;
    Tensor keep_prob_tensor(dout->type());
    keep_prob_tensor.mutable_data<T>({1}, ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&keep_prob_tensor, static_cast<T>(keep_prob));

    const auto& runner_dropout_grad = NpuOpRunner(
        "DropOutDoMask", {*dout, *mask, keep_prob_tensor}, {*dx}, {});
    runner_dropout_grad.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    dropout, ops::DropoutNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::DropoutNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    dropout_grad,
    ops::DropoutGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::DropoutGradNPUKernel<paddle::platform::NPUDeviceContext,
                              paddle::platform::float16>);

