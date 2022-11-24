// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = phi::DenseTensor;

template <typename T>
void LabelSmoothMuls(const platform::Place& place,
                     const aclrtStream& stream,
<<<<<<< HEAD
                     const Tensor* in,
                     float val,
                     Tensor* out) {
=======
                     const phi::DenseTensor* in,
                     float val,
                     phi::DenseTensor* out) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  out->mutable_data<T>(in->dims(), place);
  const auto& runner = NpuOpRunner("Muls", {*in}, {*out}, {{"value", val}});
  runner.Run(stream);
}

template <typename T>
void LabelSmoothAdds(const platform::Place& place,
                     const aclrtStream& stream,
<<<<<<< HEAD
                     const Tensor* in,
                     float val,
                     Tensor* out) {
=======
                     const phi::DenseTensor* in,
                     float val,
                     phi::DenseTensor* out) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  out->mutable_data<T>(in->dims(), place);
  const auto& runner = NpuOpRunner("Adds", {*in}, {*out}, {{"value", val}});
  runner.Run(stream);
}

template <typename T>
void LabelSmoothAddBroadCast(const platform::Place& place,
                             const aclrtStream& stream,
<<<<<<< HEAD
                             const Tensor* in1,
                             const Tensor* in2,
                             Tensor* out) {
=======
                             const phi::DenseTensor* in1,
                             const phi::DenseTensor* in2,
                             phi::DenseTensor* out) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  out->mutable_data<T>(place);
  const auto& runner = NpuOpRunner("AddV2", {*in1, *in2}, {*out}, {});
  runner.Run(stream);
}

template <typename T>
class LabelSmoothNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_t = ctx.Output<LoDTensor>("Out");
    auto* in_t = ctx.Input<LoDTensor>("X");
    auto* dist_t = ctx.Input<phi::DenseTensor>("PriorDist");
    auto epsilon = ctx.Attr<float>("epsilon");

    auto label_dim = in_t->dims()[in_t->dims().size() - 1];
    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dist_t) {
      Tensor tmp;
      Tensor dist;
      Tensor tmp2;
      LabelSmoothMuls<T>(place, stream, in_t, (1 - epsilon), &tmp);
      LabelSmoothMuls<T>(place, stream, dist_t, epsilon, &tmp2);
      tmp2.Resize({1, label_dim});
      LabelSmoothAddBroadCast<T>(place, stream, &tmp, &tmp2, out_t);
    } else {
      Tensor tmp;
      LabelSmoothMuls<T>(place, stream, in_t, (1 - epsilon), &tmp);
      LabelSmoothAdds<T>(place, stream, &tmp, (epsilon / label_dim), out_t);
    }
  }
};

template <typename T>
class LabelSmoothGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out_t = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* d_in_t = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto epsilon = ctx.Attr<float>("epsilon");

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    LabelSmoothMuls<T>(place, stream, d_out_t, 1 - epsilon, d_in_t);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(label_smooth,
                       ops::LabelSmoothNPUKernel<float>,
                       ops::LabelSmoothNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(label_smooth_grad,
                       ops::LabelSmoothGradNPUKernel<float>,
                       ops::LabelSmoothGradNPUKernel<plat::float16>);
