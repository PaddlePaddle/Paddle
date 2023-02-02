// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/fc_op.h"

namespace paddle {
namespace operators {

class FusedFCMaker : public FCOpMaker {
 protected:
  void Apply() override {
    AddAttr<bool>("use_mkldnn", "Used to schedule oneDNN kernel")
        .SetDefault(true);
    AddAttr<std::string>("mkldnn_data_type", "oneDNN operator data type")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddInput("ResidualData",
             "Extra input from fc_elementwise_add_mkldnn_fuse_pass")
        .AsDispensable()
        .AsExtra();
    AddAttr<std::string>("fuse_activation",
                         "Activation type from fc_act_mkldnn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha",
                   "Activation alpha from fc_act_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fuse_beta", "Activation beta from fc_act_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fused_output_scale",
                   "Output scale from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>(
        "fused_reshape2_shape",
        "Reshape's shape attribute from operator_reshape2_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<float>("Scale_in", "FC Input quantization scale").SetDefault(1.0f);
    AddAttr<std::vector<float>>("Scale_weights", "FC W quantization scale")
        .SetDefault({1.0f});
    AddAttr<std::vector<float>>("Bias_scales", "FC Bias quantization scales")
        .SetDefault({});
    AddAttr<float>("Scale_in_eltwise", "FC ResidualData quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_out", "FC output quantization scale")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "Flag determining if output should be converted to FP32")
        .SetDefault(false);
    AddComment(R"DOC(FC extended with oneDNN-specific fusion logic.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fused_fc,
    ops::FCOp,
    ops::FusedFCMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
