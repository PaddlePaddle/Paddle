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
class BroadcastPrimOp : public framework::OperatorBase {
 public:
  BroadcastPrimOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator broadcast_p should not be excuted directly"));
  }
};

class BroadcastPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of broadcast_p op.");
    AddOutput("Y", "(Tensor), The output tensor of broadcast_p op.");
    AddAttr<std::vector<int64_t>>(
        "shape",
        "(std::vector<int64_t>) Target shape of broadcast_p operator.");
    AddComment(R"DOC(
Autograd primitive broadcast_p operator.
)DOC");
  }
};

static void CheckShapeValid(const std::vector<int64_t> &x_shape,
                            const std::vector<int64_t> &target_shape) {
  size_t x_rank = x_shape.size();
  size_t target_rank = target_shape.size();
  PADDLE_ENFORCE_GE(target_rank, x_rank,
                    platform::errors::InvalidArgument(
                        "The rank of target shape should be greater than or "
                        "equal to input tensor's dimensions, "
                        "but received %d and %d",
                        target_rank, x_rank));
  std::vector<int64_t>::const_iterator it = target_shape.begin();
  for (size_t i = 0; i < x_rank; i++, it++) {
    if (x_shape[i] != 1) {
      it = std::find(it, target_shape.end(), x_shape[i]);
    }
    PADDLE_ENFORCE_EQ(
        it != target_shape.end(), true,
        platform::errors::InvalidArgument(
            "Invalid shape, can not broadcast input tensor into target shape,"
            "the first dismatching shape  %d is shape of input tensor at "
            "dimension %d",
            x_shape[i], i));
  }
}

class BroadcastPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto target_shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    CheckShapeValid(x_shape, target_shape);
    BOOST_GET(framework::VarDesc *, y_var_ptr)->SetShape(target_shape);
  }
};

class BroadcastPrimOpVarTypeInference
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

REGISTER_OPERATOR(broadcast_p, paddle::operators::BroadcastPrimOp,
                  paddle::operators::BroadcastPrimOpMaker,
                  paddle::operators::BroadcastPrimOpShapeInference,
                  paddle::operators::BroadcastPrimOpVarTypeInference);
