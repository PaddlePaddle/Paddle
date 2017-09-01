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

#include <random>
#include <type_traits>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

// It seems that Eigen::LODTensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T>
class CPUUniformRandomKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::LODTensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed =
        static_cast<unsigned int>(context.op_.GetAttr<int>("seed"));
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(
        static_cast<T>(context.op_.GetAttr<float>("min")),
        static_cast<T>(context.op_.GetAttr<float>("max")));
    ssize_t size = framework::product(tensor->dims());
    for (ssize_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
  }
};

class UniformRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE(GetAttr<float>("min") < GetAttr<float>("max"),
                   "uniform_random's min must less then max");
    auto* tensor = ctx.Output<framework::LODTensor>("Out");
    auto dims = GetAttr<std::vector<int>>("dims");
    tensor->Resize(framework::make_ddim(dims));
  }
};

class UniformRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  UniformRandomOpMaker(framework::OpProto* proto,
                       framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
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

REGISTER_OP_WITHOUT_GRADIENT(uniform_random, paddle::operators::UniformRandomOp,
                             paddle::operators::UniformRandomOpMaker);
REGISTER_OP_CPU_KERNEL(uniform_random,
                       paddle::operators::CPUUniformRandomKernel<float>);
