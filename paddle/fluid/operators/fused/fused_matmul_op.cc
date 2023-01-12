//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/matmul_v2_op.h"

namespace paddle {
namespace operators {

class FusedMatmulOpMaker : public MatMulV2OpMaker {
 protected:
  void Apply() override {
    AddInput("ResidualData",
             "Extra input from matmul_elementwise_add_mkldnn_fuse_pass")
        .AsDispensable()
        .AsExtra();
    AddAttr<float>("matmul_alpha", "Output scale used in matmul_v1")
        .SetDefault(1.0f);
    AddAttr<std::string>(
        "fuse_activation",
        "Activation type from matmul_activation_mkldnn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha",
                   "Activation alpha from matmul_activation_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fuse_beta",
                   "Activation beta from matmul_activation_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fused_output_scale",
                   "Output scale from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>("fused_reshape_X",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_X",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Out",
                              "Reshape's shape attribute from "
                              "matmul_transpose_reshape_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Out",
                              "Transpose's axis attribute from "
                              "matmul_transpose_reshape_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::string>("mkldnn_data_type", "oneDNN operator data type")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddAttr<float>("Scale_x", "Matmul X input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_y", "Matmul Y input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_in_eltwise", "Matmul ResidualData quantization scale")
        .SetDefault(0.0f);
    AddAttr<float>("Scale_out", "Matmul output quantization scale")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "Flag determining if output should be converted to FP32")
        .SetDefault(false);
    AddComment(
        R"DOC(Matrix multiplication extended with oneDNN-specific fusion logic.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_matmul,
    ops::MatMulV2Op,
    ops::FusedMatmulOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
