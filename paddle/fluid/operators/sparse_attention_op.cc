//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
    AddInput("X", "The input tensors of sparse_attention operator.")
        .AsDuplicable();
    AddOutput("Out", "The output tensor of sparse_attention operator");
    AddComment(
        R"DOC(Matrix multiplication Out = X * Y. A has shape (d0, d1 ... M, K), 
        B has shape (d0, d1 ... K, N), Out has shape ((d0, d1 ... M, N)). 
        In addition, it also follows the broadcast rule which is similar as
        numpy.matmul.
)DOC");
  }
};

/**
 * @brief compute the output shape and check the input shape valid or not
 */
inline framework::DDim CheckTensorShape(
    const std::vector<framework::DDim>& inputs_dims) {
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

class SparseAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "sparse_attention");

    auto inputs_dims = ctx->GetInputsDim("X");
    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_EQ(
        inputs_num, 5,
        "The  number of input tensors in sparse_attention op should be 5");
    auto out_dims = CheckTensorShape(inputs_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 1;
    for (auto* input : inputs) {
      if (!input->IsInitialized() || input->numel() == 0) {
        flag = 0;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "A input tensor of sparse_attention OP are Empty!"));
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_attention, ops::SparseAttentionOp,
                  ops::SparseAttentionOpMaker);

REGISTER_OP_CPU_KERNEL(
    sparse_attention,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext,
                               paddle::platform::complex<float>>,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext,
                               paddle::platform::complex<double>>);
