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

#include "paddle/fluid/operators/detection/prior_box_op.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

template <typename T>
void PrintTensor(const paddle::framework::Tensor& src,
                 const paddle::framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  VLOG(3) << "======= printing ======== ";
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < 10; ++i) {  // static_cast<int>(vec.size());
    VLOG(3) << "vec[" << i << "] : " << vec[i];
  }
}

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class PriorBoxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "======== PriorBoxNPUKernel begin ========";
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");

    VLOG(3) << "======== input dims  ========" << input->dims();
    VLOG(3) << "======== image dims  ========" << image->dims();

    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto place = ctx.GetPlace();
    boxes->mutable_data<T>(place);

    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");
    vars->mutable_data<T>(place);

    // Tensor tmp_out(input->type());  // Temporary Tensor
    // tmp_out.Resize(framework::make_ddim({1, 2, 32*32*12*4, 1}));
    // tmp_out.mutable_data<T>(place);

    VLOG(3) << "======== vars output dims  ========" << vars->dims();
    VLOG(3) << "======== PriorBoxNPUKernel 1 ========";
    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");  // required
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");  // required
    auto input_aspect_ratio =
        ctx.Attr<std::vector<float>>("aspect_ratios");  // required

    VLOG(3) << "======== PriorBoxNPUKernel 2 ========";
    // optional
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");
    // auto min_max_aspect_ratios_order =
    //     ctx.Attr<bool>("min_max_aspect_ratios_order");

    VLOG(3) << "======== PriorBoxNPUKernel 3 ========";
    std::vector<float> aspect_ratios;
    ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

    auto img_h = static_cast<int>(image->dims()[2]);
    auto img_w = static_cast<int>(image->dims()[3]);

    auto step_w = static_cast<T>(ctx.Attr<float>("step_w"));
    auto step_h = static_cast<T>(ctx.Attr<float>("step_h"));
    auto offset = static_cast<T>(ctx.Attr<float>("offset"));

    VLOG(3) << "======== PriorBoxNPUKernel 4 ========";

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<T>(img_w) / feature_width;
      step_height = static_cast<T>(img_h) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();  // aspect_ratios
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }
    VLOG(3) << "======== num_priors size() ======== " << num_priors;

    Tensor tmp_out(input->type());  // Temporary Tensor
    tmp_out.Resize(framework::make_ddim(
        {1, 2, feature_width * feature_height * num_priors * 4, 1}));
    tmp_out.mutable_data<T>(place);

    VLOG(3) << "======== PriorBoxNPUKernel 5 ========";
    VLOG(3) << "======== boxes output:" << boxes->dims();
    VLOG(3) << "======== PriorBoxNPUKernel 6 ========";

    VLOG(3) << "======== input h " << input->dims()[2];
    VLOG(3) << "======== input w " << input->dims()[3];

    NpuOpRunner runner;
    runner.SetType("PriorBox")
        .AddInput(*input)
        .AddInput(*image)
        .AddOutput(tmp_out)
        .AddAttr("min_size", std::vector<float>{min_sizes})
        .AddAttr("max_size", std::vector<float>{max_sizes})
        .AddAttr("aspect_ratio",
                 std::vector<float>{aspect_ratios})  // aspect_ratios
        .AddAttr("img_h", std::vector<int>{img_h})
        .AddAttr("img_w", std::vector<int>{img_w})
        .AddAttr("step_h", std::vector<float>{step_height})
        .AddAttr("step_w", std::vector<float>{step_width})
        .AddAttr("flip", std::vector<bool>{flip})
        .AddAttr("clip", std::vector<bool>{clip})
        .AddAttr("offset", std::vector<float>{offset})
        .AddAttr("variance", std::vector<float>{variances});

    // const auto& runner = NpuOpRunner("PriorBox", {*input, *image}, {*boxes},
    // {attr_input});

    VLOG(3) << "======== PriorBoxNPUKernel 7 ========";
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    VLOG(3) << "======== PriorBoxNPUKernel 8 ========";
    runner.Run(stream);

    tmp_out.Resize(framework::make_ddim(
        {2, feature_width, feature_height, num_priors, 4}));

    VLOG(3) << "========== tmp_out Slice(0,1).dims() "
            << tmp_out.Slice(0, 1).dims();
    VLOG(3) << "========== tmp_out Slice(1,2).dims() "
            << tmp_out.Slice(1, 2).dims();

    paddle::framework::TensorCopy(
        tmp_out.Slice(0, 1), place,
        ctx.template device_context<platform::DeviceContext>(), boxes);
    ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();

    paddle::framework::TensorCopy(
        tmp_out.Slice(1, 2), place,
        ctx.template device_context<platform::DeviceContext>(), vars);
    ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();

    boxes->Resize(boxes->dims());
    vars->Resize(vars->dims());

    VLOG(3) << "-------------------------打印 boxes 的值 前 10 个 "
               "=======================";
    PrintTensor<T>(*boxes, ctx);
    VLOG(3) << "-------------------------打印 vars 的值 前 10 个 "
               "=======================";
    PrintTensor<T>(*vars, ctx);

    // boxes->Resize(boxes->dims());
    // vars->Resize(vars->dims());
    VLOG(3) << "======== PriorBoxNPUKernel end ========";
    VLOG(3) << "======== boxes output:" << boxes->dims();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    prior_box,
    ops::PriorBoxNPUKernel<paddle::platform::NPUDeviceContext, float>);
