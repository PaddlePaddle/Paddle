//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sparse_attention_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class SparseAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensors of sparse_mat operator.").AsDuplicable();
    AddOutput("Out", "The output tensor of sparse_mat operator");
    AddOutput("ResultSdd", "The computation result of sdd operation.")
        .AsIntermediate();
    AddOutput("ResultSoftmax", "The computation result of softmax operation.")
        .AsIntermediate();
    AddComment(R"DOC(
      )DOC");
  }
};

/**
 * @brief compute the output shape and check the input shape valid or not
 */
inline framework::DDim CheckTensorShape(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims) {
  auto query_dim = inputs_dims[0];
  auto key_dim = inputs_dims[1];
  auto value_dim = inputs_dims[2];
  auto offset_dim = inputs_dims[3];
  auto columns_dim = inputs_dims[4];

  // query, key, value must be 4 dims
  PADDLE_ENFORCE_EQ(query_dim.size(), 4,
                    "ShapeError: Dimension in query' shapes of "
                    "sparse_attention OP must be 4. ");
  PADDLE_ENFORCE_EQ(key_dim.size(), 4,
                    "ShapeError: Dimension in key' shapes of sparse_attention "
                    "OP must be 4. ");
  PADDLE_ENFORCE_EQ(value_dim.size(), 4,
                    "ShapeError: Dimension in value' shapes of "
                    "sparse_attention OP must be 4. ");
  PADDLE_ENFORCE_EQ(offset_dim.size(), 1,
                    "ShapeError: Dimension in offset' shapes of "
                    "sparse_attention OP must be 1. ");
  PADDLE_ENFORCE_EQ(columns_dim.size(), 1,
                    "ShapeError: Dimension in columns' shapes of "
                    "sparse_attention OP must be 1. ");

  framework::DDim out_dim;

  std::vector<int64_t> new_dims(query_dim.size());
  for (int i = 0; i < query_dim.size(); i++) {
    new_dims[i] = query_dim[i];
  }

  out_dim = framework::make_ddim(new_dims);

  return out_dim;
}

/**
 * @brief compute the result shape
 */
inline framework::DDim GetResultShape(
    const std::vector<framework::DDim>& inputs_dims) {
  // auto query_dim = inputs_dims[0];
  auto columns_dim = inputs_dims[4];

  framework::DDim result_dim;

  std::vector<int64_t> new_dims;
  int64_t nnz_num = columns_dim[0];
  // new_dims.push_back(query_dim[0]*query_dim[1]);
  new_dims.push_back(nnz_num);

  result_dim = framework::make_ddim(new_dims);

  return result_dim;
}

class SparseAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("ResultSdd"), "Output", "ResultSdd",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("ResultSoftmax"), "Output", "ResultSoftmax",
                   "sparse_attention");

    auto inputs_dims = ctx->GetInputsDim("X");
    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_EQ(
        inputs_num, static_cast<size_t>(5),
        "The  number of input tensors in sparse_attention op should be 5");

    auto out_dims = CheckTensorShape(ctx->IsRuntime(), inputs_dims);
    auto result_dims = GetResultShape(inputs_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("ResultSdd", result_dims);
    ctx->SetOutputDim("ResultSoftmax", result_dims);
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    for (auto* input : inputs) {
      if (!input->IsInitialized()) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The inputs of sparse_attention OP have Empty input tensor!"));
        break;
      }
    }
    input_data_type = inputs[0]->type();

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class SparseAttentionOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInputs("X"), "Input", "X", "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("ResultSdd"), "Input", "ResultSdd",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("ResultSoftmax"), "Input", "ResultSoftmax",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "sparse_attention");

    auto in_x = "X";
    auto in_x_grad = framework::GradVarName(in_x);
    auto ins_dims = context->GetInputsDim(in_x);
    context->SetOutputsDim(in_x_grad, ins_dims);
    context->ShareAllLoD(in_x, in_x_grad);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class SparseAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sparse_attention_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ResultSdd", this->Output("ResultSdd"));
    op->SetInput("ResultSoftmax", this->Output("ResultSoftmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
  }
};

template <typename T>
class SparseAttentionDoubleGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("sparse_attention");
    grad_op->SetInput("X", this->Input(("X")));
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetOutput("DDx", this->OutputGrad(framework::GradVarName("X")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sparse_attention, ops::SparseAttentionOp,
                  ops::SparseAttentionOpMaker,
                  ops::SparseAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::SparseAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    sparse_attention_grad, ops::SparseAttentionOpGrad,
    ops::SparseAttentionDoubleGradOpMaker<paddle::framework::OpDesc>,
    ops::SparseAttentionDoubleGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    sparse_attention,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    sparse_attention_grad,
    ops::SparseAttentionGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SparseAttentionGradKernel<paddle::platform::CPUDeviceContext, double>);
