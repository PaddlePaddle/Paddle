/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/crop_tensor_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class CropTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of Op(crop_tensor) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of Op(crop_tensor) should not be null.");
    auto x_dim = ctx->GetInputDim("X");
    auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
    auto offsets = ctx->Attrs().Get<std::vector<int>>("offsets");
    if (ctx->HasInputs("ShapeTensor")) {
      // top prority shape
      auto inputs_name = ctx->Inputs("ShapeTensor");
      PADDLE_ENFORCE_GT(
          inputs_name.size(), 0,
          "Input(ShapeTensor)'size of Op(crop_tensor) can't be zero. "
          "Please check the Attr(shape)'s size of "
          "Op(fluid.layers.crop_tensor).");
      auto out_dims = std::vector<int>(inputs_name.size(), -1);
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 0) {
          out_dims[i] = static_cast<int64_t>(shape[i]);
        } else {
          if (shape[i] == -1 && offsets[i] != -1 && x_dim[i] != -1) {
            out_dims[i] = x_dim[i] - static_cast<int64_t>(offsets[i]);
          }
        }
      }
      ctx->SetOutputDim("Out", framework::make_ddim(out_dims));

      return;
    }

    if (ctx->HasInput("Shape")) {
      auto shape_dim = ctx->GetInputDim("Shape");
      PADDLE_ENFORCE_EQ(
          shape_dim.size(), 1,
          "Input(Shape)'s dimension size of Op(crop_tensor) must be 1. "
          "Please check the Attr(shape)'s dimension size of "
          "Op(fluid.layers.crop_tensor).");
      PADDLE_ENFORCE_EQ(shape_dim[0], x_dim.size(),
                        "Input(Shape)'s size of Op(crop_tensor) must be equal "
                        "to dimension size of input tensor. "
                        "Please check the Attr(shape)'s size of "
                        "Op(fluid.layers.crop_tensor).");
      if (ctx->IsRuntime()) {
        // If true, set the shape of Output(Out) according to Input(Shape) in
        // CropTensorKernel with ExecutionContext. Also check LoD in
        // CropTensorKernel.
        ctx->ShareLoD("X", /*->*/ "Out");
      } else {
        auto out_dims = std::vector<int>(shape_dim[0], -1);
        ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
      }
      return;
    }
    PADDLE_ENFORCE_EQ(int64_t(shape.size()), x_dim.size(),
                      "Attr(shape)'size of Op(crop_tensor) should be equal to "
                      "dimension size of input tensor.");
    std::vector<int64_t> out_shape(shape.size(), -1);
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] > 0) {
        out_shape[i] = static_cast<int64_t>(shape[i]);
      } else {
        if (shape[i] == -1 && offsets[i] != -1 && x_dim[i] != -1) {
          out_shape[i] = x_dim[i] - static_cast<int64_t>(offsets[i]);
        }
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensor" || var_name == "OffsetsTensor" ||
        var_name == "Shape" || var_name == "Offsets") {
      return expected_kernel_type;
    }

    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class CropTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7).");
    AddInput("Shape",
             "The input used to describe shape of output, which is a "
             "1-D vector whose size equals to the rank of input 'X'. The "
             "elements data type must be int. It has a higher priority than "
             "the shape attribute")
        .AsDispensable();
    AddInput("Offsets",
             "The input used to describe offsets in runtime, which is a "
             "1-D vector whose size equals to the rank of input 'X'. The "
             "elements data type must be int. It has a higher priority than "
             "the offsets attribute")
        .AsDispensable();
    AddInput("ShapeTensor",
             "(vector<Tensor<int32>>, optional). If provided, crop_tensor will "
             "use this. The shape of the tensor in vector MUST BE [1]. "
             "It has the highest priority compare with Input(Shape) and "
             "attr(shape).")
        .AsDuplicable()
        .AsDispensable();
    AddInput("OffsetsTensor",
             "(vector<Tensor<int32>>, optional). If provided, crop_tensor will "
             "use this. The shape of the tensor in vector MUST BE [1]. "
             "It has the highest priority compare with Input(Offsets) and "
             "attr(offsets).")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "The output of crop_tensor op, "
              "which is of the same dimensions as X.");
    AddAttr<std::vector<int>>("offsets",
                              "A list<int> describing offsets to be cropped. "
                              "The size of offsets list should be the same as "
                              "the dimension size of input X.")
        .SetDefault(std::vector<int>());
    AddAttr<std::vector<int>>("shape",
                              "A list<int> describing the shape of output. "
                              "The size of shape list should be the same as "
                              "the dimension size of input X.")
        .SetDefault(std::vector<int>());
    AddComment(R"DOC(
CropTensor Operator.

Crop input into output, as specified by offsets and shape.

There are three ways to set the offsets:
1. Input 'OffsetsTensor: It is a tensor list. It should be set as a list that 
                         contains tensor variable in python configure script. 
                         This way is suitable for dynamic offsets.
2. Input 'Offsets': It is a variable and can be output of other operators. 
                    This way is suitable for dynamic offsets.
3. Attribute 'offsets': It will be set in python configure script. This way 
                        is suitable for fixed offsets.

You CANNOT use these three ways at the same time. An exception will be raised 
if input 'OffsetsTensor' or 'Offset' is configured and meanwhile the attribute 'offsets' is 
not empty.

There are three ways to set shape:
1. Input 'ShapeTensor': It is a tensor list. It should be set as a list that contains
                        tensor variable in python configure script. This way is suitable 
                        for dynamic shape.
2. Input 'Shape': It is a Variable and can be output of other operators. This way is suitable 
                  for dynamic shape.
2. Attribute 'shape': crop input X into the shape described by a list<int>. The size of shape 
                      list should be the same as the dimension size of input X. This way is 
                      suitable for fixed shape.

The input should be a k-D tensor(k > 0 and k < 7). As an example:

Case 1:
Given

    X = [[0, 1, 2, 0, 0]
         [0, 3, 4, 0, 0]
         [0, 0, 0, 0, 0]],

and

    offsets = [0, 1],

and

    shape = [2, 2],

we get:

    Out = [[1, 2],
           [3, 4]].


Case 2:
Given

    X = [[0, 1, 2, 5, 0]
         [0, 3, 4, 6, 0]
         [0, 0, 0, 0, 0]],

and offsets is a list that contains tensor variable,
in runtime offses_var' s value is 1.

    offsets = [0, offsets_var],

and shape is a list that contains tensor variable,
in runtime dim's value is 2.

    shape = [dim, 3]

we get:

    Out = [[1, 2, 5],
           [3, 4, 6]].
)DOC");
  }
};

class CropTensorOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of Op(crop_tensor) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) of Op(crop_tensor) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensor" || var_name == "OffsetsTensor" ||
        var_name == "Shape" || var_name == "Offsets") {
      return expected_kernel_type;
    }

    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class CropTensorGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("crop_tensor_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("OffsetsTensor")) {
      op->SetInput("OffsetsTensor", this->Input("OffsetsTensor"));
    }
    if (this->HasInput("Offsets")) {
      op->SetInput("Offsets", this->Input("Offsets"));
    }
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(crop_tensor, ops::CropTensorOp, ops::CropTensorOpMaker,
                  ops::CropTensorGradOpMaker<paddle::framework::OpDesc>,
                  ops::CropTensorGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(crop_tensor_grad, ops::CropTensorOpGrad);
REGISTER_OP_CPU_KERNEL(
    crop_tensor,
    ops::CropTensorKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CropTensorKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CropTensorKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CropTensorKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    crop_tensor_grad,
    ops::CropTensorGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CropTensorGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CropTensorGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CropTensorGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
