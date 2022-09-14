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
class ReduceSumPrimOp : public framework::OperatorBase {
 public:
  ReduceSumPrimOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator reduce_sum_p should not be excuted directly"));
  }
};

class ReduceSumPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of reduce_sum_p op.");
    AddOutput("Y", "(Tensor), The output tensor of reduce_sum_p op.");
    AddAttr<std::vector<int64_t>>(
        "axis",
        "(std::vector<int64_t>) The axis along which to reduce on. Must be in "
        "range [-rank(input), rank(input)]. If `axis[i] < 0`, the axis[i] to "
        "reduce is `rank + axis[i]`.");
    AddAttr<bool>("keepdim",
                  "(bool, default false) "
                  "If true, retain the reduced axis with length 1.")
        .SetDefault(false);
    AddComment(R"DOC(
Autograd primitive reduce_sum_p operator.
)DOC");
  }
};

class ReduceSumPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = PADDLE_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto axis = ctx->Attrs().Get<std::vector<int64_t>>("axis");
    auto keepdim = ctx->Attrs().Get<bool>("keepdim");
    if (keepdim) {
      for (size_t i = 0; i < axis.size(); ++i) {
        x_shape[axis[i]] = 1;
      }
    } else {
      const int kDelFlag = -2;
      for (size_t i = 0; i < axis.size(); ++i) {
        x_shape[axis[i]] = kDelFlag;
      }
      x_shape.erase(remove(x_shape.begin(), x_shape.end(), kDelFlag),
                    x_shape.end());
    }
    if (!keepdim && x_shape.size() == 0) {
      x_shape.push_back(1);
    }

    PADDLE_GET(framework::VarDesc *, y_var_ptr)->SetShape(x_shape);
  }
};

class ReduceSumPrimOpVarTypeInference
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

REGISTER_OPERATOR(reduce_sum_p,
                  paddle::operators::ReduceSumPrimOp,
                  paddle::operators::ReduceSumPrimOpMaker,
                  paddle::operators::ReduceSumPrimOpShapeInference,
                  paddle::operators::ReduceSumPrimOpVarTypeInference);
