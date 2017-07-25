/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/random_op.h"
#include "glog/logging.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class RandomOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor*>& inputs,
      const std::vector<framework::Tensor*>& outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 0, "Input size of RandomOp must be zero.");
    PADDLE_ENFORCE(outputs.size() == 1, "Output size of RandomOp must be one.");
    PADDLE_ENFORCE(outputs[0] != nullptr,
                   "Outputs of RandomOp must all be set.");
    outputs[0]->Resize(
        framework::make_ddim(this->GetAttr<std::vector<int>>("shape")));
  }
};

class RandomOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  RandomOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<std::vector<int>>("shape", "The shape of matrix to be randomized");
    AddAttr<float>("seed", "random seed generator.").SetDefault(1337);
    AddAttr<float>("mean", "mean value of random.").SetDefault(.0);
    AddAttr<float>("std", "minimum value of random value")
        .SetDefault(1.0)
        .LargerThan(.0);
    AddOutput("Out", "output matrix of random op");
    AddComment(R"DOC(
Random Operator fill a matrix in normal distribution.
The eqution : Out = Random(Shape=(d0, d1, ...), Dtype, mean, std)
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(random,
            paddle::operators::RandomOp,
            paddle::operators::RandomOpMaker);

typedef paddle::operators::RandomOpKernel<paddle::platform::CPUPlace, float>
    RandomOpKernel_CPU_float;
REGISTER_OP_CPU_KERNEL(random, RandomOpKernel_CPU_float);
