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

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

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

    if (dropout_prob == 1.) {
      const auto& runner_zeros_out = NpuOpRunner("ZerosLike", {*out}, {*out});
      runner_zeros_out.Run(stream);
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
        // DropOutDoMask will get error result when input
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
        paddle::framework::TensorToVector(*seed_tensor, ctx.device_context(),
                                          &seed_data);
        seed = seed_data[0];
      } else {
        seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : 0;
      }

      Tensor keep_prob_tensor(x->type());
      keep_prob_tensor.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&keep_prob_tensor,
                                   static_cast<T>(keep_prob));

      mask->mutable_data<uint8_t>(ctx.GetPlace());

      // mask used in `DropOutGenMask` NPU OP is different from
      // the output `Mask`.
      Tensor npu_mask(framework::proto::VarType::UINT8);
      uint32_t length = (x->numel() + 128 - 1) / 128 * 128;
      npu_mask.Resize(framework::make_ddim({length / 8}));
      npu_mask.mutable_data<uint8_t>(ctx.GetPlace());

      // TODO(pangyoki): `keep_prob` used in `DropOutGenMask` NPU
      // OP must be a scalar with shape[0]. At present, the shape
      // of the `prob` Tensor of this OP is forced to be set to 0
      // in `npu_op_runner.cc`, which needs to be optimized later.
      NpuOpRunner runner_gen_mask;
      runner_gen_mask.SetType("DropOutGenMask")
          .AddInput(framework::vectorize(tmp_out.dims()))
          .AddInput(keep_prob_tensor)
          .AddOutput(npu_mask)
          .AddAttr("seed", seed)
          .AddAttr("seed2", seed2);
      runner_gen_mask.Run(stream);

      NpuOpRunner runner_dropout;
      runner_dropout.SetType("DropOutDoMask")
          .AddInput(tmp_x)
          .AddInput(npu_mask)
          .AddInput(keep_prob_tensor)
          .AddOutput(tmp_out);
      runner_dropout.Run(stream);

      // cast `out` from float/float16 to bool
      Tensor cast_mask(framework::proto::VarType::BOOL);
      cast_mask.Resize(mask->dims());
      cast_mask.mutable_data<bool>(ctx.GetPlace());
      auto dst_dtype_bool = ConvertToNpuDtype(cast_mask.type());
      const auto& runner_cast_mask_bool =
          NpuOpRunner("Cast", {*out}, {cast_mask},
                      {{"dst_type", static_cast<int>(dst_dtype_bool)}});
      runner_cast_mask_bool.Run(stream);

      // cast cast_mask from bool to uint8
      auto dst_dtype_uint8 = ConvertToNpuDtype(mask->type());
      const auto& runner_cast_mask_uint8 =
          NpuOpRunner("Cast", {cast_mask}, {*mask},
                      {{"dst_type", static_cast<int>(dst_dtype_uint8)}});
      runner_cast_mask_uint8.Run(stream);
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

    // cast mask from uint8 to float32/float16
    Tensor cast_mask(dx->type());
    cast_mask.Resize(mask->dims());
    cast_mask.mutable_data<T>(ctx.GetPlace());
    auto dst_dtype = ConvertToNpuDtype(dx->type());
    const auto& runner_cast_mask =
        NpuOpRunner("Cast", {*mask}, {cast_mask},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_mask.Run(stream);

    const auto& runner =
        NpuOpRunner("MaskedScale", {*dout, cast_mask}, {*dx},
                    {{"value", static_cast<float>(1. / (1 - dropout_prob))}});
    runner.Run(stream);
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
