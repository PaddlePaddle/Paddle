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

#include "paddle/fluid/operators/roi_align_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ROIAlignNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");              // (B,C,H,W）
    auto* ROIs = ctx.Input<framework::Tensor>("ROIs");        // (N，4）
    auto* ROIsNum = ctx.Input<framework::Tensor>("RoisNum");  // [0 1 1 2 2 2]
    auto* Out = ctx.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(ctx.GetPlace());

    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto sample_num = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");
    auto roi_end_mode = 0;
    PADDLE_ENFORCE_EQ(
        aligned, False,
        platform::errors::InvalidArgument(
            "ROIAlignNPU only support Aligned attribute equaled to False"));

    framework::NPUAttributeMap attr_roi = {{"spatial_scale", spatial_scale},
                                           {"pooled_height", pooled_height},
                                           {"pooled_width", pooled_width},
                                           {"sample_num", sample_num},
                                           {"roi_end_mode", roi_end_mode}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // Combine *ROIsNum with ROIs to get new ROIs
    // change roisnum's datatype & resize
    int dtype =
        static_cast<int>(ConvertToNpuDtype(framework::proto::VarType::FP32));
    framework::NPUAttributeMap attr_cast = {{"dst_type", dtype}};
    Tensor ROIsNum_fp(ROIs->type());
    ROIsNum_fp.Resize(framework::make_ddim({ROIs->dims()[0], 1}));
    ROIsNum_fp.mutable_data<T>(ctx.GetPlace());

    const auto& runner_c =
        NpuOpRunner("Cast", {*ROIsNum}, {ROIsNum_fp}, attr_cast);
    runner_c.Run(stream);

    // concate to make (N, 5)
    std::vector<paddle::framework::Tensor> x_list;
    x_list.push_back(ROIsNum_fp);
    x_list.push_back(*ROIs);
    auto axis = 1;
    // output of concate
    Tensor ROIs_N5(ROIs->type());
    ROIs_N5.Resize(framework::make_ddim({ROIs->dims()[0], 5}));
    ROIs_N5.mutable_data<T>(ctx.GetPlace());

    // attribute of concate
    auto EleNum = 2;
    framework::NPUAttributeMap attr_concat = {{"N", EleNum},
                                              {"concat_dim", axis}};

    NpuOpRunner runner0;
    runner0.SetType("ConcatD")
        .AddInputs(x_list)
        .AddOutput(ROIs_N5)
        .AddInputNames({"x0", "x1"})
        .AddAttrs(attr_concat);
    runner0.Run(stream);

    const auto& runner =
        NpuOpRunner("ROIAlign", {*X, ROIs_N5}, {*Out}, attr_roi);
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    roi_align,
    ops::ROIAlignNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ROIAlignNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::ROIAlignNPUKernel<paddle::platform::NPUDeviceContext, int>);
