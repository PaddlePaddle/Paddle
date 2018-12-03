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

class DataTransformOp : public framework::OperatorBase {
 public:
  DataTransformOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *x_var = scope.FindVar(Input("X"));
    auto *out_var = scope.FindVar(Output("Out"));

    PADDLE_ENFORCE_NOT_NULL(
        x_var, "Input(X) of data_transform op should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, "Output(Out) of data_transform op should not be null.");

    auto x_tensor = x_var->Get<framework::LoDTensor>();
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

    if (platform::is_gpu_place(x_tensor.place())) {
      // get device context from pool
      auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);

      out_tensor->Resize(x_tensor.dims());
      out_tensor->mutable_data(platform::CPUPlace(), x_tensor.type());
      framework::TensorCopy(x_tensor, platform::CPUPlace(), *dev_ctx,
                            out_tensor);
    }
    out_tensor->set_lod(x_tensor.lod());
  }
};

class DataTransformOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of data_transform op");
    AddOutput("Out", "The output of data_transform op");
    AddComment(R"DOC(
Data Transform Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(data_transform, paddle::operators::DataTransformOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::DataTransformOpMaker);
