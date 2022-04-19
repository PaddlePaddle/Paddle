// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class MatmulPrimOp : public framework::OperatorBase {
 public:
  MatmulPrimOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator matmul_p should not be excuted directly"));
  }
};

class MatmulPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of matmul_p op.");
    AddInput("Y", "(Tensor), The input tensor of matmul_p op.");
    AddOutput("Z", "(Tensor), The output tensor of matmul_p op.");
    AddComment(R"DOC(
Autograd primitive matmul_p operator.
)DOC");
  }
};

class MatmulPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetInputVarPtrs("Y")[0];
    framework::InferShapeVarPtr z_var_ptr = ctx->GetOutputVarPtrs("Z")[0];

    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    framework::VarDesc *y_var = BOOST_GET(framework::VarDesc *, y_var_ptr);
    auto x_shape = x_var->GetShape();
    auto y_shape = y_var->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();
    PADDLE_ENFORCE_EQ(x_rank, y_rank,
                      platform::errors::InvalidArgument(
                          "The two input tensor's dimension should be equal"
                          "But received first input tensor's dimension is %d, "
                          "and another input tensor's dimension is %d",
                          x_rank, y_rank));

    PADDLE_ENFORCE_EQ(x_rank == 2 || x_rank == 3, true,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension should be 2 or 3"
                          "But received input tensor's dimension is %d",
                          x_rank));

    PADDLE_ENFORCE_EQ(
        x_shape[x_rank - 1], y_shape[y_rank - 2],
        platform::errors::InvalidArgument(
            "Invalid shape for matmul, the last dimension of first input and "
            "the penultimate dimension for the second input should be same."
            "But received  %d and %d.",
            x_shape[x_rank - 1], y_shape[y_rank - 2]));
    if (x_rank == 2) {
      std::vector<int64_t> z_shape{x_shape[x_rank - 2], y_shape[y_rank - 1]};
      BOOST_GET(framework::VarDesc *, z_var_ptr)->SetShape(z_shape);
    } else {
      PADDLE_ENFORCE_EQ(x_shape[0], y_shape[0],
                        platform::errors::InvalidArgument(
                            "Invalid shape for matmul when input tensor's "
                            "dimension is 3, the first dimension of first "
                            "input and the second input should be same."
                            "But received  %d and %d.",
                            x_shape[0], y_shape[0]));

      std::vector<int64_t> z_shape{x_shape[0], x_shape[x_rank - 2],
                                   y_shape[y_rank - 1]};
      BOOST_GET(framework::VarDesc *, z_var_ptr)->SetShape(z_shape);
    }
  }
};

class MatmulPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_name = Input(ctx, "Y")[0];
    auto z_name = Output(ctx, "Z")[0];
    auto x_type = GetType(ctx, x_name);
    auto y_type = GetType(ctx, y_name);
    auto x_dtype = GetDataType(ctx, x_name);
    auto y_dtype = GetDataType(ctx, y_name);
    PADDLE_ENFORCE_EQ(x_type, y_type,
                      platform::errors::InvalidArgument(
                          "The type of two input tensor should be same, "
                          "but get %d and %d",
                          x_type, y_type));
    PADDLE_ENFORCE_EQ(x_dtype, y_dtype,
                      platform::errors::InvalidArgument(
                          "The datatype of two input tensor should be same, "
                          "but get %d and %d",
                          x_dtype, y_dtype));

    SetType(ctx, z_name, x_type);
    SetDataType(ctx, z_name, x_dtype);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(matmul_p, paddle::operators::MatmulPrimOp,
                  paddle::operators::MatmulPrimOpMaker,
                  paddle::operators::MatmulPrimOpShapeInference,
                  paddle::operators::MatmulPrimOpVarTypeInference);
