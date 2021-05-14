//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/increment_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {
class OpDesc;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class IncrementalNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    float step = context.Attr<float>("step");
    out->mutable_data<T>(context.GetPlace());

    Tensor step_tensor(x->type());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    step_tensor.mutable_data<T>({1}, context.GetPlace());
    FillNpuTensorWithConstant<T>(&step_tensor, static_cast<T>(step));

    // NOTE(zhiqiu): Why cast?  I found int64 is not supported in cann-5.0.2
    Tensor cast_x(x->type());
    Tensor cast_out(x->type());
    if (x->type() == framework::proto::VarType::INT64) {
      cast_x.Resize(x->dims());
      cast_x.mutable_data<int>(ctx.GetPlace());
      cast_out.Resize(out->dims());
      cast_out.mutable_data<int>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(cast_x->type());
      auto runner_cast_x = NpuOpRunner(
          "Cast", {*x}, {cast_x}, {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_x.Run(stream);
    } else {
      cast_x.ShareDataWith(*x);
      cast_out.ShareDataWith(*out);
    }
    auto runner = NpuOpRunner("Add", {*cast_x, step_tensor}, {*cast_out}, {});

    if (x->type() == framework::proto::VarType::INT64) {
      auto dst_dtype = ConvertToNpuDtype(out->type());
      auto runner_cast_out =
          NpuOpRunner("Cast", {cast_out}, {*out},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_out.Run(stream);
    }

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    increment,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::IncrementalNPUKernel<paddle::platform::NPUDeviceContext,
                              plat::float16>)
