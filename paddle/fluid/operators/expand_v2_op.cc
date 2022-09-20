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

#include "paddle/fluid/operators/expand_v2_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "expand_shapes_tensor" || var_name == "Shape") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class ExpandV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddInput("Shape",
             "(Tensor<int>), optional). If provided, expand according to "
             "this given Shape. It has a higher priority than "
             "expand_shapes_tensor and the shape attribute.")
        .AsDispensable();
    AddInput("expand_shapes_tensor",
             "(Tensor Tensor<int>), epxanded shape for X."
             "It has a higher priority than shape attribute, but a lower "
             "priority than the input Shape")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddAttr<std::vector<int>>("shape", "The expanded shape for each dimension.")
        .SetDefault({});
    AddComment(R"DOC(
Expand the input to the given shape. The rank of X
should be in [1, 6] and size of 'shape' must be in [1, 6] also.
Following is a using case:

Input(X) is a 3-D tensor with shape [2, 3, 1]:

        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]

Attr(shape):  [2, 6, 2]

Output(Out) is a 3-D tensor with shape [2, 6, 2]:

        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]

)DOC");
  }
};

class ExpandV2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandV2Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "ExpandV2Grad");

    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> expand_shape = ctx->Attrs().Get<std::vector<int>>("shape");
    if (expand_shape.size() == 0) {
      expand_shape = std::vector<int>(x_dims.size(), -1);
    }

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dim_vec = phi::vectorize<int>(x_dims);
    auto diff = expand_shape.size() - x_dim_vec.size();
    x_dim_vec.insert(x_dim_vec.begin(), diff, -1);

    for (size_t i = 0; i < expand_shape.size(); ++i) {
      if (expand_shape[i] < 0 || x_dim_vec[i] == -1) {
        continue;
      } else {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(
              expand_shape[i],
              out_dims[i],
              platform::errors::InvalidArgument(
                  "The size (%d) of the dimension %d of Input(Out@GRAD) should "
                  "be equal to the crroresponding dimension size of shape(%d).",
                  out_dims[i],
                  i,
                  expand_shape[i]));
        }
      }
    }
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "expand_shapes_tensor" || var_name == "Shape") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

template <typename T>
class ExpandV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_v2_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("expand_shapes_tensor", this->Input("expand_shapes_tensor"));
    op->SetInput("Shape", this->Input("Shape"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ExpandV2DoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_v2");
    op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    if (this->HasInput("expand_shapes_tensor")) {
      op->SetInput("expand_shapes_tensor", this->Input("expand_shapes_tensor"));
    }
    if (this->HasInput("Shape")) {
      op->SetInput("Shape", this->Input("Shape"));
    }
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandV2GradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(expand_v2,
                            ExpandInferShapeFunctor,
                            PD_INFER_META(phi::ExpandInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand_v2,
                  ops::ExpandV2Op,
                  ops::ExpandV2OpMaker,
                  ops::ExpandV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandV2GradOpMaker<paddle::imperative::OpBase>,
                  ExpandInferShapeFunctor);
REGISTER_OPERATOR(expand_v2_grad,
                  ops::ExpandV2GradOp,
                  ops::ExpandV2DoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandV2DoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::ExpandV2GradNoNeedBufVarsInferer);
