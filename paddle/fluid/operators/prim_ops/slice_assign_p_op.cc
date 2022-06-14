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
class SliceAssignPrimOp : public framework::OperatorBase {
 public:
  SliceAssignPrimOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator slice_assign_p should not be excuted directly"));
  }
};

class SliceAssignPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The tensor to slice from and assign on.");
    AddInput("Y", "(Tensor), The source tensor of slice_assign_p op.");
    AddOutput("Z", "(Tensor), The output tensor of slice_assign_p op.");
    AddAttr<std::vector<int64_t>>(
        "axis", "(std::vector<int64_t>), The axis along which to gather.");
    AddAttr<std::vector<int64_t>>(
        "starts",
        "(std::vector<int64_t>) The slice starts of slice_assign_p op");
    AddAttr<std::vector<int64_t>>(
        "ends", "(std::vector<int64_t>) The slice ends of slice_assign_p op");
    AddAttr<std::vector<int64_t>>(
        "strides",
        "(std::vector<int64_t>) The slice strides of slice_assign_p op");
    AddComment(R"DOC(
Autograd primitive slice_assign_p operator.
)DOC");
  }
};

class SliceAssignPrimOpShapeInference : public framework::InferShapeBase {
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
    PADDLE_ENFORCE_EQ(x_rank, y_rank,
                      platform::errors::InvalidArgument(
                          "The dimensions of two input tensor should be same, "
                          "but get %d and %d",
                          x_rank, y_rank));
    std::vector<int64_t> y_target_shape(x_shape);
    for (size_t i = 0; i < axis.size(); ++i) {
      y_target_shape[axis[i]] =
          (ends[i] - starts[i] + strides[i] - 1) / strides[i];
    }
    for (size_t i = 0; i < x_rank; ++i) {
      PADDLE_ENFORCE_EQ(y_target_shape[i], y_shape[i],
                        platform::errors::InvalidArgument(
                            "The shape of source tensor of slice_assign_p op "
                            "at dimension %d should be %d, "
                            "but get %d",
                            i, y_target_shape[i], y_shape[i]));
    }
    BOOST_GET(framework::VarDesc *, z_var_ptr)->SetShape(x_shape);
  }
};

class SliceAssignPrimOpVarTypeInference
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

    SetType(ctx, z_name, GetType(ctx, x_name));
    SetDataType(ctx, z_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(slice_assign_p, paddle::operators::SliceAssignPrimOp,
                  paddle::operators::SliceAssignPrimOpMaker,
                  paddle::operators::SliceAssignPrimOpShapeInference,
                  paddle::operators::SliceAssignPrimOpVarTypeInference);
