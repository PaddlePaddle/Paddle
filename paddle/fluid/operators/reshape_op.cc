/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/reshape_op.h"

namespace paddle {
namespace operators {

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReshapeOp should not be null.");

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE(!shape.empty(),
                   "The shape information must be set by Attr(shape).");

    std::vector<int64_t> output_shape;
    auto x_dims = ctx->GetInputDim("X");
    bool need_copy_dim = ValidateShape(shape, x_dims, output_shape);

    if (need_copy_dim) {
      // Some dimensions can only be determined during runtime. Here temporarily
      // set output tensor's shape the same as that of the input tensor.
      ctx->SetOutputDim("Out", x_dims);
    } else {
      ctx->SetOutputDim("Out", framework::make_ddim(output_shape));

      // FIXME(caoying): When shape of the output tensor is determined during
      // runtime, LoD information of X will not passed to the output.
      if (shape[0] == x_dims[0]) {
        // Only pass LoD when the first dimension of output and Input(X)
        // are the same.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

 private:
  bool ValidateShape(const std::vector<int> &shape,
                     const framework::DDim &input_dim,
                     std::vector<int64_t> &output_shape) const {
    // only one dimension canbe set to -1, whose size will be automatically
    // infered.
    const int64_t unknown_index = -1;
    const auto in_size = framework::product(input_dim);
    const auto x_rank = input_dim.size();

    bool need_dim_copy = false;
    std::vector<size_t> neg_dims_idx;
    for (size_t i = 0; i < shape.size(); ++i) {
      PADDLE_ENFORCE(shape[i] >= 0 || shape[i] == unknown_index,
                     "Each input dimension of Attr(shape) must be positive, or "
                     "only one input dimension can be -1.");
      if (shape[i] == unknown_index) {
        neg_dims_idx.push_back(i);
      } else if (shape[i] == 0) {
        PADDLE_ENFORCE_LT(
            i, x_rank,
            "Only dimension less than rank of Input(X) can be set to 0.");
        need_dim_copy = true;
      }
    }
    PADDLE_ENFORCE_LE(
        neg_dims_idx.size(), 1,
        "Only one input dimension of Attr(shape) may be unknown.");

    output_shape.resize(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), output_shape.begin(),
                   [](int a) { return static_cast<int64_t>(a); });

    // some dimension can only be determinted during runtime.
    if (need_dim_copy) return need_dim_copy;

    int64_t inferred_dim = 0;
    if (neg_dims_idx.size()) {
      int64_t capacity = std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int>());
      inferred_dim = in_size / (-capacity);
      PADDLE_ENFORCE_EQ(inferred_dim * (-capacity), in_size,
                        "Invalid shape is given.");
      output_shape[neg_dims_idx[0]] = inferred_dim;
    }
    return false;
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReshapeOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of reshape operator.");
    AddOutput("Out", "The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.");
    AddAttr<bool>("inplace",
                  "Change the source tensor's shape without copy memory.")
        .SetDefault(true);
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape).

An example:
Given a 2-D tensor X with 2 rows and 2 columns : [[1, 2], [3, 4]]

and target shape = [1, 4], the reshape operator will transform
the tensor X into a 2-D tensor: [[1, 2, 3, 4]]

One dimension in the target shape can be set -1, representing that its
size is unknown. In this case, the real dimension will be infered from
the original shape of Input(X) and other dimensions in the target shape.
)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OP(reshape, ops::ReshapeOp, ops::ReshapeOpMaker, reshape_grad,
            ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL(reshape, ops::ReshapeKernel<CPU, float>,
                       ops::ReshapeKernel<CPU, double>,
                       ops::ReshapeKernel<CPU, int>,
                       ops::ReshapeKernel<CPU, int64_t>);
REGISTER_OP_CPU_KERNEL(reshape_grad, ops::ReshapeGradKernel<CPU, float>,
                       ops::ReshapeGradKernel<CPU, double>,
                       ops::ReshapeGradKernel<CPU, int>,
                       ops::ReshapeGradKernel<CPU, int64_t>);
