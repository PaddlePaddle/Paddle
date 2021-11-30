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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static void CumsumImp(const Tensor& input, Tensor* output,
                      const framework::NPUAttributeMap& attr_input,
                      const framework::ExecutionContext& ctx) {
  auto stream =
      ctx.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();
  if (input.type() == framework::proto::VarType::INT64) {
    Tensor tmp_input;
    tmp_input.mutable_data<float>(input.dims(), ctx.GetPlace());
    auto dst_acl_dtype = ConvertToNpuDtype(tmp_input.type());
    const auto& cast_runner_1 =
        NpuOpRunner("Cast", {input}, {tmp_input},
                    {{"dst_type", static_cast<int>(dst_acl_dtype)}});
    cast_runner_1.Run(stream);

    Tensor tmp_output;
    tmp_output.mutable_data<float>(output->dims(), ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("CumsumD", {tmp_input}, {tmp_output}, attr_input);
    runner.Run(stream);

    dst_acl_dtype = ConvertToNpuDtype(output->type());
    const auto& cast_runner_2 =
        NpuOpRunner("Cast", {tmp_output}, {*output},
                    {{"dst_type", static_cast<int>(dst_acl_dtype)}});
    cast_runner_2.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("CumsumD", {input}, {*output}, attr_input);
    runner.Run(stream);
  }
}

template <typename DeviceContext, typename T>
class CumSumNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool reverse = ctx.Attr<bool>("reverse");

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {
        {"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}};

    bool flatten = ctx.Attr<bool>("flatten");
    if (flatten) {
      PADDLE_ENFORCE_EQ(
          axis, -1,
          platform::errors::InvalidArgument(
              "when flatten is true, attr axis must be default %d, but got %d",
              -1, axis));

      Tensor new_x(x->type());
      new_x.ShareDataWith(*x);

      new_x.Resize(framework::make_ddim({x->numel()}));

      CumsumImp(new_x, out, attr_input, ctx);
    } else {
      CumsumImp(*x, out, attr_input, ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    cumsum, ops::CumSumNPUKernel<plat::NPUDeviceContext, int>,
#ifdef PADDLE_WITH_ASCEND_INT64
    ops::CumSumNPUKernel<plat::NPUDeviceContext, int64_t>,
#endif
    ops::CumSumNPUKernel<plat::NPUDeviceContext, float>,
    ops::CumSumNPUKernel<plat::NPUDeviceContext, plat::float16>);
