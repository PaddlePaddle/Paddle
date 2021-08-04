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

#include "paddle/fluid/operators/norm_op.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
void print_matrix(const framework::ExecutionContext& ctx, const Tensor& t) {
  std::vector<T> bad_out_data(t.numel());
  framework::TensorToVector<T>(
      t, ctx.template device_context<paddle::platform::NPUDeviceContext>(),
      &bad_out_data);
  std::string ret = "";
  for (int i = 0; i < t.numel(); ++i) {
    ret += std::to_string(bad_out_data[i]) + " ";
  }
  VLOG(4) << t.dims() << "DATA: \n" << ret << std::endl;
}

template <typename DeviceContext, typename T>
class NormNPUKernel : public framework::OpKernel<T> {
  void CheckAxis(int axis, int rank) const {
    // check the axis is in [-rank, rank-1]
    if (axis <= rank - 1 && axis >= -rank) return;
    PADDLE_ENFORCE_EQ(0, 1,
                      platform::errors::InvalidArgument(
                          "axis in norm operator must between (%d) and (%d)"
                          "but (%d) got.",
                          -rank, rank - 1, axis));
  }

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(4) << "Norm NPU Kernel" << std::endl;
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_y = ctx.Output<framework::Tensor>("Out");
    auto* out_norm = ctx.Output<framework::Tensor>("Norm");
    out_y->mutable_data<T>(ctx.GetPlace());
    out_norm->mutable_data<T>(ctx.GetPlace());
    auto xdim = in_x->dims();
    float eps = static_cast<float>(ctx.Attr<float>("epsilon"));
    int axis = ctx.Attr<int>("axis");
    CheckAxis(axis, xdim.size());
    if (axis < 0) axis = xdim.size() + axis;

    framework::NPUAttributeMap attr_input_norm;
    attr_input_norm["axes"] = std::vector<int>({axis});
    attr_input_norm["p"] = 2;
    attr_input_norm["keepdim"] = true;
    attr_input_norm["epsilon"] = eps;
    const auto& runner =
        NpuOpRunner("LpNorm", {*in_x}, {*out_norm}, attr_input_norm);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
    // the out_norm have the true value
    auto in_x_shape = framework::vectorize<int32_t>(xdim);

    Tensor norm_tensor_bc(out_norm->type());
    norm_tensor_bc.mutable_data<T>(xdim, ctx.GetPlace());

    /*  ERROR   //   error, output size mismatch
        Tensor shape_tensor(framework::proto::VarType::INT32);
        framework::TensorFromVector<int32_t>(in_x_shape, ctx.device_context(),
       &shape_tensor);
        const auto & runner_bc = NpuOpRunner("BroadcastTo", {*out_norm,
       shape_tensor}, {norm_tensor_bc}, {});
    */
    /*  Use BroadcastToD is Ok*/
    const auto& runner_bc = NpuOpRunner(
        "BroadcastToD", {*out_norm}, {norm_tensor_bc}, {{"shape", in_x_shape}});

    runner_bc.Run(stream);
    NpuOpRunner("Div", {*in_x, norm_tensor_bc}, {*out_y}, {}).Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    norm, ops::NormNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NormNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>)
