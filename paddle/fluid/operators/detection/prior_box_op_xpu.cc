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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/detection/prior_box_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename T, typename K>
class PriorBoxOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    auto input_aspect_ratio = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");
    auto min_max_aspect_ratios_order =
        ctx.Attr<bool>("min_max_aspect_ratios_order");

    std::vector<float> aspect_ratios;
    ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

    K step_w = static_cast<K>(ctx.Attr<float>("step_w"));
    K step_h = static_cast<K>(ctx.Attr<float>("step_h"));
    K offset = static_cast<K>(ctx.Attr<float>("offset"));

    auto img_width = image->dims()[3];
    auto img_height = image->dims()[2];

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    K step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<K>(img_width) / feature_width;
      step_height = static_cast<K>(img_height) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }

    boxes->mutable_data<K>(ctx.GetPlace());
    vars->mutable_data<K>(ctx.GetPlace());

    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    auto boxes_data = boxes->data<K>();
    auto vars_data = vars->data<K>();
    xpu::VectorParam<float> aspect_ratios_param{
        aspect_ratios.data(), static_cast<int>(aspect_ratios.size()), nullptr};
    xpu::VectorParam<float> min_sizes_param{
        min_sizes.data(), static_cast<int>(min_sizes.size()), nullptr};
    xpu::VectorParam<float> max_sizes_param{
        max_sizes.data(), static_cast<int>(max_sizes.size()), nullptr};

    int ret = xpu::gen_prior_box(
        dev_ctx.x_context(), boxes_data, aspect_ratios_param, min_sizes_param,
        max_sizes_param, feature_height, feature_width, img_height, img_width,
        offset, step_height, step_width, clip, min_max_aspect_ratios_order);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gen_prior_box");

    int box_num = feature_height * feature_width * num_priors;
    int vlen = variances.size();
    std::vector<K> var_cpu(vlen * box_num);
    for (int i = 0; i < box_num; ++i) {
      std::copy(variances.begin(), variances.end(), var_cpu.begin() + i * vlen);
    }
    ret = xpu_memcpy(vars_data, var_cpu.data(), var_cpu.size() * sizeof(K),
                     XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    PADDLE_ENFORCE_XPU_SUCCESS(ret);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(prior_box, ops::PriorBoxOpXPUKernel<float, float>);

#endif
