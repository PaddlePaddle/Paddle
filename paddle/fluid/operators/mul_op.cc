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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

constexpr int kMULMKLDNNINT8 = 1;
constexpr int kMULMKLDNNFP32 = 2;

class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;

      if (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
          input_data_type == framework::DataTypeTrait<uint8_t>::DataType()) {
        customized_type_value = kMULMKLDNNINT8;
      } else if (input_data_type ==
                     framework::DataTypeTrait<
                         paddle::platform::bfloat16>::DataType() ||
                 input_data_type ==
                     framework::DataTypeTrait<float>::DataType()) {
        customized_type_value = kMULMKLDNNFP32;
      }
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of mul op.");
    AddInput("Y", "(Tensor), The second input tensor of mul op.");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<float>(
        "scale_x",
        "scale_x to be used for int8 mul input data x. scale_x has the"
        "same purpose as scale_in in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<std::vector<float>>(
        "scale_y",
        "scale_y to be used for int8 mul input data y. scale_y has the"
        "same purpose as scale_weights in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault({1.0f})
        .AsExtra();
    AddAttr<float>("scale_out",
                   "scale_out to be used for int8 output data."
                   "Only used with MKL-DNN INT8")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<bool>(
        "force_fp32_output",
        "(bool, default false) Force quantize kernel output FP32, only "
        "used in quantized MKL-DNN.")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Mul Operator.

This operator is used to perform matrix multiplication for input $X$ and $Y$.

The equation is:

$$Out = X * Y$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class MulOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class MulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;

      if (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
          input_data_type == framework::DataTypeTrait<uint8_t>::DataType()) {
        customized_type_value = kMULMKLDNNINT8;
      } else if (input_data_type ==
                     framework::DataTypeTrait<
                         paddle::platform::bfloat16>::DataType() ||
                 input_data_type ==
                     framework::DataTypeTrait<float>::DataType()) {
        customized_type_value = kMULMKLDNNFP32;
      }
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

template <typename T>
class MulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mul_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

class MulDoubleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mul");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "mul");
    OP_INOUT_CHECK(ctx->HasInput("DOut"), "Input", "DOut", "mul");

    if (ctx->HasOutput("DDOut") &&
        (ctx->HasInput("DDX") || (ctx->HasInput("DDY")))) {
      ctx->ShareDim("DOut", "DDOut");
    }
    if (ctx->HasOutput("DX") && ctx->HasInput("DDY")) {
      ctx->ShareDim("X", "DX");
    }
    if (ctx->HasOutput("DY") && ctx->HasInput("DDX")) {
      ctx->ShareDim("Y", "DY");
    }
  }
};

template <typename T>
class MulDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mul_grad_grad");

    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    retv->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    retv->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    auto ddx = this->OutputGrad(framework::GradVarName("X"));
    auto ddw = this->OutputGrad(framework::GradVarName("Y"));

    if (!ddx.empty() || !ddw.empty()) {
      retv->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    }
    retv->SetOutput(
        "DX", ddw.empty() ? this->EmptyInputGrad() : this->InputGrad("X"));
    retv->SetOutput(
        "DY", ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Y"));

    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(mul, MulInferShapeFunctor,
                            PD_INFER_META(phi::MatmulWithFlattenInferMeta));
REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker, ops::MulOpInferVarType,
                  ops::MulOpGradMaker<paddle::framework::OpDesc>,
                  ops::MulOpGradMaker<paddle::imperative::OpBase>,
                  MulInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(mul_grad, MulGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralBinaryGradInferMeta));
REGISTER_OPERATOR(mul_grad, ops::MulGradOp,
                  ops::MulDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MulDoubleGradMaker<paddle::imperative::OpBase>,
                  MulGradInferShapeFunctor);

REGISTER_OPERATOR(mul_grad_grad, ops::MulDoubleGradOp);
