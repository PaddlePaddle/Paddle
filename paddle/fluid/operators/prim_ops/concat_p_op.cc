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
class ConcatPrimOp : public framework::OperatorBase {
 public:
  ConcatPrimOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator concat_p should not be excuted directly"));
  }
};

class ConcatPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("XS", "(Tensor), The input tensors of concat_p op.");
    AddOutput("Y", "(Tensor), The output tensor of concat_p op.");
    AddAttr<int64_t>("axis", "(int64_t), The axis along which to concat.");
  }
};

class ConcatPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    auto x_var_ptrs = ctx->GetOutputVarPtrs("XS");
    PADDLE_ENFORCE_GT(
        x_var_ptrs.size(), 0,
        platform::errors::InvalidArgument(
            "Number of"
            " input tensors of concat_p op should be at least 1"));
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    auto axis = ctx->Attrs().Get<int64_t>("axis");
    int64_t cnt_along_axis = 0;
    framework::VarDesc *first_x_var =
        BOOST_GET(framework::VarDesc *, x_var_ptrs[0]);
    auto first_x_shape = first_x_var->GetShape();
    cnt_along_axis += first_x_shape[axis];
    for (size_t i = 1; i < x_var_ptrs.size(); ++i) {
      framework::VarDesc *x_var =
          BOOST_GET(framework::VarDesc *, x_var_ptrs[i]);
      auto x_shape = x_var->GetShape();
      cnt_along_axis += x_shape[axis];
      // TODO(lml): add some check for x_shape
    }

    std::vector<int64_t> y_shape(first_x_shape);
    y_shape[axis] = cnt_along_axis;
    BOOST_GET(framework::VarDesc *, y_var_ptr)->SetShape(y_shape);
  }
};

class ConcatPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    // TODO(lml): add some check for multi input
    auto first_x_name = Input(ctx, "XS")[0];
    auto y_name = Output(ctx, "Y")[0];
    SetType(ctx, y_name, GetType(ctx, first_x_name));
    SetDataType(ctx, y_name, GetDataType(ctx, first_x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(concat_p, paddle::operators::ConcatPrimOp,
                  paddle::operators::ConcatPrimOpShapeInference,
                  paddle::operators::ConcatPrimOpVarTypeInference);
