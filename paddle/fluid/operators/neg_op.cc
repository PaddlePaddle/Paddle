// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace operators {

class NegOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of neg op.");
    AddOutput("Out", "(Tensor), The output tensor of neg op.");
		AddComment(R"DOC(
Neg Operator.

This operator is used to perform elementwise neg for input $X$.
$$out = |x|$$

)DOC");
	}
};

template <typename T>
class NegGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("neg_grad");
		retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
		retv->SetAttrMap(this->Attrs());
		retv->SetOutput(framework::GradVarName("X"), this->InputputGrad("X"));
	}
};


}  // namespace operators
}  // namespace paddle
