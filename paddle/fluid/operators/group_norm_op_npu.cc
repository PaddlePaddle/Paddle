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

#include "paddle/fluid/operators/group_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

#include "paddle/fluid/framework/tensor_util.h"
template <typename T>
void PrintTensor(const framework::Tensor& src,
                 const framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < static_cast<int>(vec.size()); ++i) {
    std::cout << "vec[" << i << "] : " << vec[i] << std::endl;
  }
}

template <typename DeviceContext, typename T>
class GroupNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    // auto* mean = ctx.Output<Tensor>("Mean");
    // auto* var = ctx.Output<Tensor>("Variance");
    const auto G = ctx.Attr<int>("groups");

    const auto x_dims = x->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int group_size = (C - 1) / G + 1;

    // check NCHW first

    // ignore scale and bias
    Tensor default_scale;
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim({C}), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(1.0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", {C}}});
      runner.Run(stream);
      scale = &default_scale;
    }

    Tensor default_bias;
    if (!bias) {
      default_bias.mutable_data<T>(framework::make_ddim({C}), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_bias}, {{"dims", {C}}});
      runner.Run(stream);
      bias = &default_bias;
    }

    // y->mutable_data<T>(place);
    // mean->mutable_data<T>(place);
    // var->mutable_data<T>(place);

    //////////////////////////////
    // call GNTrainingReduce first
    Tensor sum_tmp, square_sum_tmp;
    sum_tmp.mutable_data<T>(framework::make_ddim({x_dims[0], G, 1, 1, 1}),
                            place);
    square_sum_tmp.mutable_data<T>(
        framework::make_ddim({x_dims[0], G, 1, 1, 1}), place);

    const auto& runner_reduce =
        NpuOpRunner("GNTrainingReduce", {*x}, {sum_tmp, square_sum_tmp},
                    {{"num_groups", G}});
    runner_reduce.Run(stream);

    std::cout << "/// x: " << std::endl;
    PrintTensor<T>(*x, ctx);
    std::cout << "/// sum: " << std::endl;
    PrintTensor<T>(sum_tmp, ctx);
    std::cout << "/// square_sum: " << std::endl;
    PrintTensor<T>(square_sum_tmp, ctx);

    /////////////////////////////
    // then call GNTrainingUpdate
    Tensor batch_mean_tmp, batch_var_tmp;
    batch_mean_tmp.mutable_data<T>(
        framework::make_ddim({x_dims[0], G, 1, 1, 1}), place);
    batch_var_tmp.mutable_data<T>(framework::make_ddim({x_dims[0], G, 1, 1, 1}),
                                  place);

    y->mutable_data<T>(place);
    y->Resize({x_dims[0], G, group_size, x_dims[2], x_dims[3]});
    // y_tmp.mutable_data<T>(framework::make_ddim({x_dims[0], G, group_size,
    // x_dims[2], x_dims[3]}), place);

    const auto& runner_update =
        NpuOpRunner("GNTrainingUpdate",
                    {*x, sum_tmp, square_sum_tmp},  // scale, offset, mean, var
                    {*y, batch_mean_tmp, batch_var_tmp},
                    {{"epsilon", epsilon}, {"num_groups", G}});
    runner_update.Run(stream);

    std::cout << "/// y[5D]: " << std::endl;
    PrintTensor<T>(*y, ctx);

    y->Resize(x_dims);
    std::cout << "/// y[4D]: " << std::endl;
    PrintTensor<T>(*y, ctx);

    /* this op is not implemented in CANN
    const auto& runner = NpuOpRunner(
        "GroupNorm", {*x, *scale, *bias}, {*y, *mean, *var},
        {{"epsilon", epsilon},
         {"data_format", (data_layout == DataLayout::kNCHW ? "NCHW" : "NHWC")},
         {"is_training", false},
         {"num_groups", G}});
    runner.Run(stream);
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    group_norm,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);
