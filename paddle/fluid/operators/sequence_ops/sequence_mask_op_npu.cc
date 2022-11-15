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

#include "paddle/fluid/operators/sequence_ops/sequence_mask_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class SequenceMaskNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    int maxlen = ctx.Attr<int>("maxlen");

    if (ctx.HasInput("MaxLenTensor")) {
      auto max_len_tensor = ctx.Input<phi::DenseTensor>("MaxLenTensor");
      PADDLE_ENFORCE_NOT_NULL(max_len_tensor,
                              platform::errors::InvalidArgument(
                                  "Input(MaxLenTensor) should not be NULL."
                                  "But received Input(MaxLenTensor) is NULL"));
      phi::DenseTensor temp;
      paddle::framework::TensorCopySync(
          *max_len_tensor, platform::CPUPlace(), &temp);
      maxlen = *temp.data<int32_t>();
      PADDLE_ENFORCE_GT(
          maxlen,
          0,
          platform::errors::InvalidArgument(
              "Input(MaxLenTensor) value should be greater than 0. But "
              "received Input(MaxLenTensor) value = %d.",
              maxlen));
    }

    if (maxlen < 0) {
      auto x_numel = x->numel();
      std::vector<T> x_vec;
      framework::TensorToVector(*x, dev_ctx, &x_vec);
      auto x_data = x_vec.data();
      maxlen = static_cast<int>(*std::max_element(x_data, x_data + x_numel));
    }
    auto y_dim = phi::vectorize<int>(x->dims());
    y_dim.push_back(maxlen);

    Tensor cast_x;
    cast_x.mutable_data<int32_t>(x->dims(), ctx.GetPlace());
    const auto& cast1_runner = NpuOpRunner(
        "Cast",
        {*x},
        {cast_x},
        {{"dst_type",
          ConvertToNpuDtype(framework::TransToProtoVarType(cast_x.dtype()))}});
    cast1_runner.Run(dev_ctx.stream());

    Tensor tmp;
    tmp.mutable_data<int32_t>(phi::make_ddim({maxlen}), ctx.GetPlace());
    NpuOpRunner range_runner;
    range_runner.SetType("Range");
    range_runner.AddInput(std::vector<int32_t>({0}));
    range_runner.AddInput(std::vector<int32_t>({maxlen}));
    range_runner.AddInput(std::vector<int32_t>({1}));
    range_runner.AddOutput(tmp);
    range_runner.Run(dev_ctx.stream());

    Tensor expand_tmp;
    expand_tmp.mutable_data<int32_t>(phi::make_ddim(y_dim), ctx.GetPlace());
    const auto& expand_runner =
        NpuOpRunner("ExpandD", {tmp}, {expand_tmp}, {{"shape", y_dim}});
    expand_runner.Run(dev_ctx.stream());

    auto x_dims = phi::vectorize<int>(x->dims());
    x_dims.push_back(1);
    cast_x.Resize(phi::make_ddim({x_dims}));
    Tensor x_tmp;
    x_tmp.mutable_data<int32_t>(phi::make_ddim(y_dim), ctx.GetPlace());
    const auto& tile_runner =
        NpuOpRunner("TileWithAxis",
                    {cast_x},
                    {x_tmp},
                    {{"axis", x->dims().size()}, {"tiles", maxlen}});
    tile_runner.Run(dev_ctx.stream());

    Tensor y_tmp;
    y_tmp.mutable_data<uint8_t>(phi::make_ddim(y_dim), ctx.GetPlace());
    const auto& less_runner =
        NpuOpRunner("Less", {expand_tmp, x_tmp}, {y_tmp}, {});
    less_runner.Run(dev_ctx.stream());

    y->Resize(phi::make_ddim(y_dim));
    auto out_dtype = static_cast<framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));
    if (out_dtype == framework::proto::VarType::INT32) {
      y->mutable_data<int32_t>(ctx.GetPlace());
    } else if (out_dtype == framework::proto::VarType::INT64) {
      y->mutable_data<int64_t>(ctx.GetPlace());
    } else if (out_dtype == framework::proto::VarType::FP32) {
      y->mutable_data<float>(ctx.GetPlace());
    } else if (out_dtype == framework::proto::VarType::FP64) {
      y->mutable_data<double>(ctx.GetPlace());
    } else if (out_dtype == framework::proto::VarType::BOOL) {
      y->mutable_data<bool>(ctx.GetPlace());
    } else if (out_dtype == framework::proto::VarType::UINT8) {
      y->mutable_data<uint8_t>(ctx.GetPlace());
    } else {
      PADDLE_ENFORCE(false,
                     platform::errors::InvalidArgument(
                         "out_dtype only supporing int32, int64, fp32, fp64, "
                         "bool, uint8, but receive out_dtype is %d",
                         out_dtype));
    }

    const auto& cast2_runner = NpuOpRunner(
        "Cast", {y_tmp}, {*y}, {{"dst_type", ConvertToNpuDtype(out_dtype)}});
    cast2_runner.Run(dev_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    sequence_mask,
    ops::SequenceMaskNPUKernel<plat::NPUDeviceContext, int32_t>,
    ops::SequenceMaskNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::SequenceMaskNPUKernel<plat::NPUDeviceContext, float>,
    ops::SequenceMaskNPUKernel<plat::NPUDeviceContext, double>);
