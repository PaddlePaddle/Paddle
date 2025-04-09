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

#include <string>
#include <vector>

#include "paddle/fluid/operators/transpose_op.h"

namespace paddle::operators {

class FusedTransposeOpMaker : public Transpose2OpMaker {
 protected:
  void Apply() override {
    AddAttr<std::vector<int>>("fused_squeeze2_axes",
                              "Axes from squeeze2 operator obtained from "
                              "squeeze2_transpose2_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_unsqueeze2_axes",
                              "Axes from unsqueeze2 operator obtained from "
                              "operator_unsqueeze2_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape2_shape",
                              "Shape from reshape2 operator obtained from "
                              "operator_reshape2_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<float>("scale",
                   "Obtained from quant_transpose2_dequant_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<float>("shift",
                   "Obtained from quant_transpose2_dequant_onednn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<std::string>(
        "output_data_type",
        "Obtained from quant_transpose2_dequant_onednn_fuse_pass")
        .SetDefault("");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fused_transpose,
    ops::Transpose2Op,
    ops::FusedTransposeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
