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

#include "paddle/operators/gaussian_random_op.h"
#include "glog/logging.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class GaussianRandomOpKernel<platform::CPUPlace, T>
    : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto mean = context.op_.GetAttr<T>("mean");
    auto std = context.op_.GetAttr<T>("std");
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();
    T* r = output->mutable_data<T>(context.GetPlace());
    auto ctx =
        static_cast<const platform::CPUDeviceContext*>(context.device_context_);
    // generator need to modify context
    auto g = const_cast<platform::CPUDeviceContext*>(ctx)->RandGenerator();
    std::normal_distribution<T> distribution(mean, std);
    for (int i = 0; i < framework::product(output->dims()); ++i) {
      r[i] = distribution(g);
    }
  }
};

class GaussianRandomOp : public framework::OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE(inputs.size() == 0, "Input size of RandomOp must be zero.");
    PADDLE_ENFORCE(outputs.size() == 1, "Output size of RandomOp must be one.");
    PADDLE_ENFORCE(outputs[0] != nullptr,
                   "Outputs of RandomOp must all be set.");
    auto* tensor = ctx.Output<Tensor>(0);
    auto dims = GetAttr(std::vector<int>("shape"));
    tensor->Resize(framework::make_ddim(dims));
  }
};

class GaussianRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GaussianRandomOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<std::vector<int>>("shape", "The shape of matrix to be randomized");
    AddAttr<float>("mean", "mean value of random.").SetDefault(.0);
    AddAttr<float>("std", "minimum value of random value")
        .SetDefault(1.0)
        .LargerThan(.0);
    AddOutput("Out", "output matrix of random op");
    AddComment(R"DOC(
GaussianRandom Operator fill a matrix in normal distribution.
The eqution : Out = GaussianRandom(Shape=(d0, d1, ...), Dtype, mean, std)
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(gaussian_random, paddle::operators::GaussianRandomOp,
            paddle::operators::GaussianRandomOpMaker);

typedef paddle::operators::GaussianRandomOpKernel<paddle::platform::CPUPlace,
                                                  float>
    GaussianRandomOpKernel_CPU_float;
REGISTER_OP_CPU_KERNEL(gaussian_random, GaussianRandomOpKernel_CPU_float);
