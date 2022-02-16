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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/pten/kernels/funcs/math_function.h"

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
        aligned, false,
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
    Tensor ROIsNum_fp(ROIs->dtype());
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
    Tensor ROIs_N5(ROIs->dtype());
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

template <typename T>
class ROIAlignNPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sample_num = ctx.Attr<int>("sampling_ratio");
    auto in_dims = in->dims();
    auto aligned = ctx.Attr<bool>("aligned");

    int rois_num = rois->dims()[0];

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (!in_grad) {
      return;
    }
    in_grad->mutable_data<T>(place);

    PADDLE_ENFORCE_EQ(
        aligned, false,
        platform::errors::InvalidArgument(
            "ROIAlignGradNPU only support Aligned attribute equaled to False"));
    PADDLE_ENFORCE_EQ(
        ctx.HasInput("RoisNum"), true,
        platform::errors::NotFound("Input(RoisNum) of ROIAlignGradOp "
                                   "is not found while using NPU."));
    PADDLE_ENFORCE_EQ(
        framework::TransToProtoVarType(rois->dtype()),
        framework::proto::VarType::FP32,
        platform::errors::InvalidArgument(
            "ROIAlignGradNPU only support ROIs type equaled to FP32."));

    // Cast RoisNum to fp32 tensor
    auto* RoisNum = ctx.Input<framework::Tensor>("RoisNum");
    Tensor ROIs_N5;
    ROIs_N5.mutable_data<float>({rois_num, 5}, place);
    Tensor ROIsNum_fp;
    ROIsNum_fp.mutable_data<T>(RoisNum->dims(), place);  // shape = [rois_num]
    int nputype_fp32 =
        static_cast<int>(ConvertToNpuDtype(framework::proto::VarType::FP32));
    const auto& runner_cast = NpuOpRunner("Cast", {*RoisNum}, {ROIsNum_fp},
                                          {{"dst_type", nputype_fp32}});
    runner_cast.Run(stream);
    ROIsNum_fp.Resize({rois_num, 1});

    // Combine *ROIsNum with ROIs to get new ROIs
    std::vector<paddle::framework::Tensor> x_list;
    x_list.push_back(ROIsNum_fp);
    x_list.push_back(*rois);
    const auto& runner_concat = NpuOpRunner("ConcatD", {x_list}, {ROIs_N5},
                                            {{"N", 2}, {"concat_dim", 1}});
    runner_concat.Run(stream);

    //  By analysis, in order to match cpu grad version,
    //  rois[:,3:5] should substrate 1 before call ascend grad function
    std::vector<float> vec_dlt = {0, 0, 0, -1.0f, -1.0f};
    Tensor tsr_dlt;
    tsr_dlt.mutable_data<float>({5}, place);
    framework::TensorFromVector<float>(vec_dlt, ctx.device_context(), &tsr_dlt);
    ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
    const auto& runner_add =
        NpuOpRunner("AddV2", {ROIs_N5, tsr_dlt}, {ROIs_N5}, {});
    runner_add.Run(stream);

    //  Call ascend RoiAlignGrad function
    int roi_end_mode = 0;
    const auto& runner_roi_align_grad =
        NpuOpRunner("ROIAlignGrad", {*out_grad, ROIs_N5}, {*in_grad},
                    {{"xdiff_shape", framework::vectorize<int>(in_dims)},
                     {"pooled_width", pooled_width},
                     {"pooled_height", pooled_height},
                     {"spatial_scale", spatial_scale},
                     {"sample_num", sample_num},
                     {"roi_end_mode", roi_end_mode}});
    runner_roi_align_grad.Run(stream);
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

REGISTER_OP_NPU_KERNEL(roi_align_grad, ops::ROIAlignNPUGradKernel<float>,
                       ops::ROIAlignNPUGradKernel<double>,
                       ops::ROIAlignNPUGradKernel<int>);
