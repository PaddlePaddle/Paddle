/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/clip_by_norm_op.h"

namespace paddle {
namespace operators {

class ClipByNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ClipByNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ClipByNormOp should not be null.");
    auto max_norm = ctx->Attrs().Get<float>("max_norm");
    PADDLE_ENFORCE_GT(max_norm, 0, "max_norm should be greater than 0.");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ClipByNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input of clip_by_norm op."
             "The number of dimensions must be between [1, 9].");
    AddOutput("Out",
              "(Tensor) The output of clip_by_norm op with shape as input(X)");
    AddAttr<float>("max_norm", "(float) The maximum norm value.");
    AddComment(R"DOC(
ClipByNorm Operator.

This operator limits the L2 norm of the input $X$ within $max\_norm$.
If the L2 norm of $X$ is less than or equal to $max\_norm$, $Out$ will be
the same as $X$. If the L2 norm of $X$ is greater than $max\_norm$, $X$ will
be linearly scaled to make the L2 norm of $Out$ equal to $max\_norm$, as
shown in the following formula:

$$
Out = \\frac{max\\_norm * X}{norm(X)},
$$

where $norm(X)$ represents the L2 norm of $X$.

Examples:
        .. code-block:: python

            data = fluid.layer.data(
                name='data', shape=[2, 4, 6], dtype='float32')
            reshaped = fluid.layers.clip_by_norm(
                x=data, max_norm=0.5)

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle
