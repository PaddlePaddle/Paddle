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
class TransposePrimOp : public framework::OperatorBase {
 public:
  TransposePrimOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator transpose_p should not be excuted directly"));
  }
};

class TransposePrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of transpose_p op.");
    AddOutput("Y", "(Tensor), The output tensor of transpose_p op.");
    AddAttr<std::vector<int64_t>>("axis",
                                  "(std::vector<int64_t>) Tanspose axis.");
    AddComment(R"DOC(
Autograd primitive transpose_p operator.
)DOC");
  }
};

class TransposePrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto axis = ctx->Attrs().Get<std::vector<int64_t>>("axis");
    size_t x_rank = x_shape.size();
    size_t axis_size = axis.size();
    PADDLE_ENFORCE_EQ(x_rank, axis_size,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension "
                          "should be equal to the axis's size. "
                          "But received input tensor's dimension is %d, "
                          "axis's size is %d",
                          x_rank, axis_size));

    std::vector<int> count(axis_size, 0);
    for (size_t i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE_GE(axis[i], 0,
                        platform::errors::InvalidArgument(
                            "The axis should be greater than or equal to 0."
                            "But received %d of axis[%d]",
                            axis[i], i));

      PADDLE_ENFORCE_EQ(
          axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1, true,
          platform::errors::InvalidArgument(
              "Each element of Attribute axis should "
              "be a unique value range from 0 to (dims - 1), "
              "where the dims is the axis's size, "
              "unique value means this axis value can appear only once. "
              "But received axis[%d] is %d, axis_size is %d, "
              "count[axis[%d]] is %d",
              i, axis[i], axis_size, i, count[axis[i]]));
    }
    std::vector<int64_t> y_shape(axis_size);
    for (size_t i = 0; i < axis_size; i++) {
      y_shape[i] = x_shape[axis[i]];
    }
    BOOST_GET(framework::VarDesc *, y_var_ptr)->SetShape(y_shape);
  }
};

class TransposePrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_name = Output(ctx, "Y")[0];
    SetType(ctx, y_name, GetType(ctx, x_name));
    SetDataType(ctx, y_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(transpose_p, paddle::operators::TransposePrimOp,
                  paddle::operators::TransposePrimOpMaker,
                  paddle::operators::TransposePrimOpShapeInference,
                  paddle::operators::TransposePrimOpVarTypeInference);
