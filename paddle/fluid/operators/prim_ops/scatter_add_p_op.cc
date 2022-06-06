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
class ScatterAddPrimOp : public framework::OperatorBase {
 public:
  ScatterAddPrimOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator scatter_add_p should not be excuted directly"));
  }
};

class ScatterAddPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The tensor to apply scatter rule and add on.");
    AddInput("Y", "(Tensor), The source tensor of scatter_add_p op.");
    AddInput(
        "IndexTensor",
        "(Tensor), The index tensor of scatter_add_p op, which is a 1D tensor.")
        .AsDispensable();
    AddOutput("Z", "(Tensor), The output tensor of scatter_add_p op.");
    AddAttr<int64_t>("axis",
                     "(int64_t), The axis along which to scatter and add.");
    AddAttr<std::vector<int64_t>>(
        "index", "(std::vector<int64_t>) The index of scatter_add_p op")
        .SetDefault({0});
    AddComment(R"DOC(
Autograd primitive scatter_add_p operator.
)DOC");
  }
};

class ScatterAddPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetInputVarPtrs("Y")[0];
    framework::InferShapeVarPtr z_var_ptr = ctx->GetOutputVarPtrs("Z")[0];
    int64_t num_index = 0;
    if (ctx->HasInput("IndexTensor")) {
      framework::InferShapeVarPtr index_var_ptr =
          ctx->GetInputVarPtrs("IndexTensor")[0];
      framework::VarDesc *index_var =
          BOOST_GET(framework::VarDesc *, index_var_ptr);
      auto index_shape = index_var->GetShape();
      PADDLE_ENFORCE_EQ(index_shape.size(), 1,
                        platform::errors::InvalidArgument(
                            "The index tensor should be a 1D tensor,"
                            "but get rank %d",
                            index_shape.size()));
      num_index = index_shape[0];
    } else {
      num_index = ctx->Attrs().Get<std::vector<int64_t>>("index").size();
    }
    auto axis = ctx->Attrs().Get<int64_t>("axis");
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    framework::VarDesc *y_var = BOOST_GET(framework::VarDesc *, y_var_ptr);
    auto x_shape = x_var->GetShape();
    auto y_shape = y_var->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();
    PADDLE_ENFORCE_EQ(x_rank, y_rank,
                      platform::errors::InvalidArgument(
                          "The dimensions of two input tensor should be same, "
                          "but get %d and %d",
                          x_rank, y_rank));
    PADDLE_ENFORCE_EQ(y_shape[axis], num_index,
                      platform::errors::InvalidArgument(
                          "The shape of source input tensor at scatter axis "
                          "should be  equal to num_index, "
                          "but get %d and %d",
                          y_shape[axis], num_index));
    for (size_t i = 0; i < x_rank; ++i) {
      if (i != size_t(axis)) {
        PADDLE_ENFORCE_EQ(
            x_shape[i], y_shape[i],
            platform::errors::InvalidArgument(
                "The shape of two input tensor at dimension %d should be same, "
                "but get %d and %d",
                i, x_rank, y_rank));
      }
    }

    BOOST_GET(framework::VarDesc *, z_var_ptr)->SetShape(x_shape);
  }
};

class ScatterAddPrimOpVarTypeInference
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

    if (ctx->HasInput("IndexTensor")) {
      auto index_name = Input(ctx, "IndexTensor")[0];
      auto index_dtype = GetDataType(ctx, index_name);
      PADDLE_ENFORCE_EQ(
          index_dtype, framework::proto::VarType_Type_INT32,
          platform::errors::InvalidArgument(
              "The datatype of input tensor should be VarType_Type_INT32(%d), "
              "but get %d",
              framework::proto::VarType_Type_INT32, index_dtype));
    }
    SetType(ctx, z_name, GetType(ctx, x_name));
    SetDataType(ctx, z_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(scatter_add_p, paddle::operators::ScatterAddPrimOp,
                  paddle::operators::ScatterAddPrimOpMaker,
                  paddle::operators::ScatterAddPrimOpShapeInference,
                  paddle::operators::ScatterAddPrimOpVarTypeInference);
