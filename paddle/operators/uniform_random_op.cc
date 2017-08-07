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

#include "paddle/operators/uniform_random_op.h"

namespace paddle {
namespace operators {
class RandomOp : public OperatorWithKernel {
 protected:
  void InferShape(const InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(GetAttr<float>("min") < GetAttr<float>("max"),
                   "uniform_random's min must less then max");
    auto tensor = ctx.Output<Tensor>(0);
    auto dims = GetAttr<std::vector<int>>("dims");
    tensor->Resize(framework::make_ddim(dims));
  }
};

class RandomOpMaker : public OpProtoAndCheckerMaker {
 public:
  RandomOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "The output tensor of uniform random op");
    AddComment(R"DOC(Uniform random operator.

Used to initialize tensor with uniform random generator.
)DOC");
    AddAttr<std::vector<int>>("dims", "the dimension of random tensor");
    AddAttr<float>("min", "Minimum value of uniform random").SetDefault(-1.0f);
    AddAttr<float>("max", "Maximun value of uniform random").SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed of uniform random. "
                 "0 means generate a seed by system")
        .SetDefault(0);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(uniform_random, ops::RandomOp, ops::RandomOpMaker);
REGISTER_OP_CPU_KERNEL(uniform_random, ops::CPUUniformRandomKernel<float>);
