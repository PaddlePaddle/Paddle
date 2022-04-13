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
class SliceSelectPrimOp : public framework::OperatorBase {
 public:
  SliceSelectPrimOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator slice_select_p should not be excuted directly"));
  }
};

class SliceSelectPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of slice_select_p op.");
    AddOutput("Y", "(Tensor), The output tensor of slice_select_p op.");
    AddAttr<std::vector<int64_t>>(
        "axis", "(std::vector<int64_t>), The axis along which to gather.");
    AddAttr<std::vector<int64_t>>(
        "starts",
        "(std::vector<int64_t>) The slice starts of slice_select_p op");
    AddAttr<std::vector<int64_t>>(
        "ends", "(std::vector<int64_t>) The slice ends of slice_select_p op");
    AddAttr<std::vector<int64_t>>(
        "strides",
        "(std::vector<int64_t>) The slice strides of slice_select_p op");
    AddComment(R"DOC(
Autograd primitive slice_select_p operator.
)DOC");
  }
};

class SliceSelectPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto axis = ctx->Attrs().Get<std::vector<int64_t>>("axis");
    auto starts = ctx->Attrs().Get<std::vector<int64_t>>("starts");
    auto ends = ctx->Attrs().Get<std::vector<int64_t>>("ends");
    auto strides = ctx->Attrs().Get<std::vector<int64_t>>("strides");
    PADDLE_ENFORCE_EQ(
        starts.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Number of starts attribute and axis attribute should be same, "
            "but get %d and %d",
            starts.size(), axis.size()));
    PADDLE_ENFORCE_EQ(
        ends.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Number of ends attribute and axis attribute should be same, "
            "but get %d and %d",
            ends.size(), axis.size()));
    PADDLE_ENFORCE_EQ(
        strides.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Number of strides attribute and axis attribute should be same, "
            "but get %d and %d",
            strides.size(), axis.size()));
    for (size_t i = 0; i < axis.size(); ++i) {
      x_shape[axis[i]] = (ends[i] - starts[i] + strides[i] - 1) / strides[i];
    }
    BOOST_GET(framework::VarDesc *, y_var_ptr)->SetShape(x_shape);
  }
};

class SliceSelectPrimOpVarTypeInference
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

REGISTER_OPERATOR(slice_select_p, paddle::operators::SliceSelectPrimOp,
                  paddle::operators::SliceSelectPrimOpMaker,
                  paddle::operators::SliceSelectPrimOpShapeInference,
                  paddle::operators::SliceSelectPrimOpVarTypeInference);
