/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/detection/prior_box_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class PriorBoxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* image = ctx.Input<phi::DenseTensor>("Image");
    auto* boxes = ctx.Output<phi::DenseTensor>("Boxes");
    auto* variances = ctx.Output<phi::DenseTensor>("Variances");
    float step_w = ctx.Attr<float>("step_w");
    float step_h = ctx.Attr<float>("step_h");
    float offset = ctx.Attr<float>("offset");
    bool clip = ctx.Attr<bool>("clip");
    bool min_max_aspect_ratios_order =
        ctx.Attr<bool>("min_max_aspect_ratios_order");

    int im_width = image->dims()[3];
    int im_height = image->dims()[2];
    int width = input->dims()[3];
    int height = input->dims()[2];

    auto aspect_ratios = ctx.Attr<std::vector<float>>("aspect_ratios");
    bool flip = ctx.Attr<bool>("flip");
    std::vector<float> new_aspect_ratios;
    ExpandAspectRatios(aspect_ratios, flip, &new_aspect_ratios);
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    phi::DenseTensor ratios;
    paddle::framework::TensorFromVector(new_aspect_ratios, dev_ctx, &ratios);
    MLUOpTensorDesc new_aspect_ratios_desc(ratios);

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    phi::DenseTensor min;
    paddle::framework::TensorFromVector(min_sizes, dev_ctx, &min);
    MLUOpTensorDesc min_sizes_desc(min);

    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    phi::DenseTensor max;
    paddle::framework::TensorFromVector(max_sizes, dev_ctx, &max);
    MLUOpTensorDesc max_sizes_desc(max);

    auto variances_attr = ctx.Attr<std::vector<float>>("variances");
    phi::DenseTensor var_tensor;
    paddle::framework::TensorFromVector(variances_attr, dev_ctx, &var_tensor);
    MLUOpTensorDesc variances_attr_desc(var_tensor);

    auto place = ctx.GetPlace();

    boxes->mutable_data<T>(place);
    variances->mutable_data<T>(place);

    MLUOpTensorDesc var_desc(*variances);
    MLUOpTensorDesc output_desc(*boxes);
    MLUOP::OpPriorBox(ctx,
                      min_sizes_desc.get(),
                      GetBasePtr(&min),
                      new_aspect_ratios_desc.get(),
                      GetBasePtr(&ratios),
                      variances_attr_desc.get(),
                      GetBasePtr(&var_tensor),
                      max_sizes_desc.get(),
                      GetBasePtr(&max),
                      height,
                      width,
                      im_height,
                      im_width,
                      step_h,
                      step_w,
                      offset,
                      clip,
                      min_max_aspect_ratios_order,
                      output_desc.get(),
                      GetBasePtr(boxes),
                      var_desc.get(),
                      GetBasePtr(variances));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(prior_box, ops::PriorBoxMLUKernel<float>);
