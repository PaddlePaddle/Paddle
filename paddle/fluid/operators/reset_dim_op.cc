// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class ResetDim : public framework::OperatorBase {
 public:
  ResetDim(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Input("Input")));
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Output("Output")));
    auto &in_tensor =
        scope.FindVar(Input("Input"))->Get<framework::LoDTensor>();
    auto out_tensor =
        scope.FindVar(Output("Output"))->GetMutable<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(&in_tensor, out_tensor);
    auto new_dim = Attr<std::vector<int>>("new_dim");
    size_t mem_size = 1;
    for (auto &ele : new_dim) {
      mem_size *= ele;
    }

    PADDLE_ENFORCE_GT(mem_size, 0);
    PADDLE_ENFORCE_LE(mem_size * framework::SizeOfType(in_tensor.type()),
                      out_tensor->memory_size());
    out_tensor->Resize(framework::make_ddim(new_dim));
  }
};

class ResetDimMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", ".");
    AddOutput("Output", "");
    AddAttr<std::vector<int>>("new_dim", ".");
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(reset_dim, paddle::operators::ResetDim,
                  paddle::operators::ResetDimMaker);
