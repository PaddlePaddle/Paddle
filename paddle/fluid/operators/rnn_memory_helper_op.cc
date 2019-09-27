/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
class RNNMemoryHelperOp : public framework::OperatorBase {
 public:
  RNNMemoryHelperOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto mem_var_name = Input("X");
    auto *mem_var = scope.FindVar(mem_var_name);
    PADDLE_ENFORCE(mem_var != nullptr,
                   "Cannot find mem_var in scope, mem_var_name is %s",
                   mem_var_name);

    auto out_name = this->Output("Out");
    auto *out_var = scope.FindVar(out_name);
    PADDLE_ENFORCE(out_var != nullptr,
                   "Cannot find out_var in scope, out_var_name is %s",
                   out_name);

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    auto &mem_tensor = mem_var->Get<framework::LoDTensor>();
    framework::TensorCopy(mem_tensor, dev_place, dev_ctx, out_tensor);
    out_tensor->set_lod(mem_tensor.lod());
  }
};

class RNNMemoryHelperOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of rnn_memory_helper op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of rnn_memory_helper op should not be null.");
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class RNNMemoryHelperOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "");
    AddOutput("Out", "");
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddComment("");
  }
};

class RNNMemoryHelperGradOp : public framework::OperatorBase {
 public:
  RNNMemoryHelperGradOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto out_grad_var_name = Input(framework::GradVarName("Out"));
    auto *out_grad_var = scope.FindVar(out_grad_var_name);

    auto in_grad_var_name = Output(framework::GradVarName("X"));
    auto *in_grad_var = scope.FindVar(in_grad_var_name);

    PADDLE_ENFORCE(in_grad_var != nullptr,
                   "Cannot find in_grad_var in scope, name is %s",
                   in_grad_var_name);

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    if (out_grad_var == nullptr) {
      VLOG(5) << "Using fill constant 0 as starting gradient";
      auto in_var_name = Input("X");
      auto *in_var = scope.FindVar(in_var_name);
      auto &in_var_tensor = in_var->Get<framework::LoDTensor>();

      framework::AttributeMap attrs;
      attrs["dtype"] = in_var_tensor.type();
      attrs["shape"] = framework::vectorize<int>(in_var_tensor.dims());
      attrs["value"] = 0.0f;

      auto zero_op = framework::OpRegistry::CreateOp(
          "fill_constant", {}, {{"Out", {in_grad_var_name}}}, attrs);
      zero_op->Run(scope, dev_place);
    } else {
      auto &out_grad_tensor = out_grad_var->Get<framework::LoDTensor>();
      auto *in_grad_tensor = in_grad_var->GetMutable<framework::LoDTensor>();
      framework::TensorCopy(out_grad_tensor, dev_place, dev_ctx,
                            in_grad_tensor);
      in_grad_tensor->set_lod(out_grad_tensor.lod());
    }
  }
};

class RNNMemoryHelperGradOpInfoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(framework::GradVarName("Out"), "");
    AddInput("X", "");
    AddInput("Out", "");
    AddOutput(framework::GradVarName("X"), "");
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddComment("");
  }
};

class RNNMemoryHelperGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    auto x_grad_name = framework::GradVarName("X");
    PADDLE_ENFORCE(ctx->HasOutput(x_grad_name),
                   "Gradient of Input(X) in rnn_memory_helper_grad of should "
                   "not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of rnn_memory_helper_grad of should not be null.");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ x_grad_name);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(rnn_memory_helper, paddle::operators::RNNMemoryHelperOp,
                  paddle::operators::RNNMemoryHelperOpInfoMaker,
                  paddle::operators::RNNMemoryHelperOpShapeInference,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(rnn_memory_helper_grad,
                  paddle::operators::RNNMemoryHelperGradOp,
                  paddle::operators::RNNMemoryHelperGradOpInfoMaker,
                  paddle::operators::RNNMemoryHelperGradOpShapeInference);
