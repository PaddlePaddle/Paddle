//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

class FusedElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInputX();
    AddInputY();
    AddOpOutput();
    AddAttr<int>("axis",
                 "(int, default -1). If X.dimension != Y.dimension,"
                 "Y.dimension must be a subsequence of x.dimension. And axis "
                 "is the start dimension index "
                 "for broadcasting Y onto X. ")
        .SetDefault(-1);

    AddAttr<std::string>(
        "fuse_activation",
        "Activation type from elementwise_act_onednn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha", "Alfa value for the elementwise operator")
        .SetDefault(1.0f);
    AddAttr<float>("fuse_beta", "Beta value for the elementwise operator")
        .SetDefault(1.0f);

    AddAttr<float>("scale_x", "Obtained from cpu_quantize_pass")
        .SetDefault(1.0f);
    AddAttr<float>("scale_y", "Obtained from cpu_quantize_pass")
        .SetDefault(1.0f);
    AddAttr<float>("scale_out", "Obtained from cpu_quantize_pass")
        .SetDefault(1.0f);

    AddAttr<float>("fused_output_scale",
                   "Obtained from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>(
        "fused_unsqueeze2_axes",
        "Obtained from operator_unsqueeze2_onednn_fuse_pass for "
        "elementwise_mul")
        .SetDefault({});
    AddOpComment();
  }

 protected:
  virtual void AddInputX() {
    AddInput("X", "(Tensor), The first input tensor of elementwise op.");
  }
  virtual void AddInputY() {
    AddInput("Y", "(Tensor), The second input tensor of elementwise op.");
  }
  virtual void AddOpOutput() {
    AddOutput("Out",
              "N-dimension tensor. A location into which the result is stored. "
              "It's dimension "
              "equals with x");
  }
  virtual void AddOpComment() { AddComment(GetCommentExamples()); }

  virtual std::string GetOpFuntionality() const { return ""; }

  virtual std::string GetName() const = 0;
  virtual std::string GetEquation() const = 0;

  std::string GetCommentExamples() const {
    return string::Sprintf(R"DOC(
Elementwise %s Operator.

%s

The equation is:

$$%s$$

- $X$: a tensor of any dimension.
- $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.

There are two cases for this operator:

1. The shape of $Y$ is the same with $X$.
2. The shape of $Y$ is a continuous subsequence of $X$.

For case 2:

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
   for broadcasting $Y$ onto $X$.
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: text

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

)DOC",
                           GetName(),
                           GetOpFuntionality(),
                           GetEquation());
  }
};

}  // namespace operators
}  // namespace paddle
