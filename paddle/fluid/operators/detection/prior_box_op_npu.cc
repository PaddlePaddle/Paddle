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
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class PriorBoxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(WARNING) << "PriorBoxNPUKernel";
    LOG(WARNING) << "op type: " << ctx.Type();

    auto* input = ctx.Input<Tensor>("Input");
    auto* image = ctx.Input<Tensor>("Image");
    auto* boxes = ctx.Output<Tensor>("Boxes");
    auto* variances = ctx.Output<Tensor>("Variances");

    // LOG(WARNING) << "input.numel: " << input->numel();
    // LOG(WARNING) << "image.numel: " << image->numel();
    // LOG(WARNING) << "boxes.numel: " << boxes->numel();
    // LOG(WARNING) << "variances.numel: " << variances->numel();

    // LOG(WARNING) << "input.dims: " << input->dims();
    // LOG(WARNING) << "image.dims: " << image->dims();
    // LOG(WARNING) << "boxes.dims: " << boxes->dims();
    // LOG(WARNING) << "variances.dims: " << variances->dims();

    PADDLE_ENFORCE_EQ(boxes->dims(), variances->dims(),
                      platform::errors::Unimplemented(
                          "the shape of boxes and variances must be same in "
                          "the npu kernel of prior_box, but got boxes->dims() "
                          "= [%s], variances->dims() = [%s]",
                          boxes->dims(), variances->dims()));

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    auto aspect_ratios = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances_attr = ctx.Attr<std::vector<float>>("variances");
    bool flip = ctx.Attr<bool>("flip");
    bool clip = ctx.Attr<bool>("clip");
    float step_w = ctx.Attr<float>("step_w");
    float step_h = ctx.Attr<float>("step_h");
    float offset = ctx.Attr<float>("offset");
    bool min_max_aspect_ratios_order =
        ctx.Attr<bool>("min_max_aspect_ratios_order");

    if (!min_max_aspect_ratios_order) {
      LOG(WARNING) << "When attr min_max_aspect_ratios_order=False, the "
                      "precision of the npu kernel of prior_box only have "
                      "atol=1e-1.";
    }

    // PADDLE_ENFORCE_EQ(
    //     min_max_aspect_ratios_order, true,
    //     platform::errors::Unimplemented(
    //         "min_max_aspect_ratios_order=False is not supported in "
    //         "the npu kernel of prior_box."));

    auto place = ctx.GetPlace();
    boxes->mutable_data<T>(place);

    Tensor out(input->type());
    // out.Resize(framework::make_ddim({2, 32, 32, 12, 4}));
    auto out_dims = framework::vectorize(boxes->dims());
    out_dims.insert(out_dims.begin(), 2);
    out.Resize(framework::make_ddim(out_dims));
    LOG(WARNING) << "out_dims: " << outputVector(out_dims);
    out.mutable_data<T>(place);

    framework::NPUAttributeMap attr_input = {{"min_size", min_sizes},
                                             {"max_size", max_sizes},
                                             {"aspect_ratio", aspect_ratios},
                                             {"step_h", step_h},
                                             {"step_w", step_w},
                                             {"flip", flip},
                                             {"clip", clip},
                                             {"offset", offset},
                                             {"variance", variances_attr}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner =
        NpuOpRunner("PriorBox", {*input, *image}, {out}, attr_input);
    // const auto& runner =
    //     NpuOpRunner("PriorBox", {*input, *image}, {*boxes}, attr_input);
    // const auto& runner =
    //     NpuOpRunner("PriorBox", {*input, *image}, {*variances}, attr_input);
    runner.Run(stream);

    out.Resize(framework::make_ddim({out.numel()}));
    // LOG(WARNING) << "here";
    Tensor out_boxes = out.Slice(0, boxes->numel());
    // LOG(WARNING) << "here";
    Tensor out_variances = out.Slice(boxes->numel(), out.numel());
    // LOG(WARNING) << "here";

    out_boxes.Resize(boxes->dims());
    out_variances.Resize(variances->dims());
    // LOG(WARNING) << "here";

    framework::TensorCopy(
        out_boxes, place,
        ctx.template device_context<platform::NPUDeviceContext>(), boxes);
    framework::TensorCopy(
        out_variances, place,
        ctx.template device_context<platform::NPUDeviceContext>(), variances);

    // LOG(WARNING) << "out_boxes: ";
    // PrintTensor<T>(out_boxes, ctx);
    // LOG(WARNING) << "out_variances: ";
    // PrintTensor<T>(out_variances, ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    prior_box, ops::PriorBoxNPUKernel<plat::NPUDeviceContext, float>,
    ops::PriorBoxNPUKernel<plat::NPUDeviceContext, plat::float16>);
