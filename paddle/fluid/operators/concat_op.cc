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

#include "paddle/fluid/operators/concat_op.h"

#include <paddle/fluid/platform/complex.h>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"

#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

#ifdef PADDLE_WITH_MKLDNN
#include <paddle/fluid/platform/mkldnn_helper.h>
#endif

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto *input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "All Inputs of Concat OP are Empty!"));
    }
#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false)
        .AsExtra();
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated."
                 "The axis could also be negative numbers. Negative axis is "
                 "interpreted as counting from the end of the rank."
                 "i.e., axis + rank(X) th dimension.")
        .SetDefault(0);
    AddInput("AxisTensor",
             "(Tensor) The axis along which the input tensors will be "
             "concatenated.  "
             "It has higher priority than Attr(axis). "
             "The shape of AxisTensor must be [1].")
        .AsDispensable();
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
    AddComment(R"DOC(
Concat Operator.

Concatenate the input tensors along dimension axis.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6]]
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));

    ctx->ShareAllLoD(in_x, out_x_g_n);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    // extra checking if attr "use_mkldnn" exist is needed because
    // test_reverse_op is calling concat_grad kernel without setting
    // "use_mkldnn" to any value
    if (ctx.HasAttr("use_mkldnn") &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ConcatOpGradNoNeedBufferVarInferer, "X");

template <typename T>
class ConcatGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("concat_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("AxisTensor")) {
      op->SetInput("AxisTensor", this->Input("AxisTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ConcatDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("concat");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(concat, ConcatInferShapeFunctor,
                            PD_INFER_META(phi::ConcatInferMeta));

REGISTER_OPERATOR(concat, ops::ConcatOp, ops::ConcatOpMaker,
                  ops::ConcatGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatGradOpMaker<paddle::imperative::OpBase>,
                  ConcatInferShapeFunctor);
REGISTER_OPERATOR(concat_grad, ops::ConcatOpGrad,
                  ops::ConcatDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConcatDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::ConcatOpGradNoNeedBufferVarInferer);
