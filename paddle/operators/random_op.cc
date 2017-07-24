#include "paddle/operators/random_op.h"
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
    PADDLE_ENFORCE(inputs[0] != nullptr && outputs[0] != nullptr,
                   "Inputs/Outputs of RandomOp must all be set.");
    outputs[0]->set_dims(inputs[0]->dims());
  }
};

class RandomOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  RandomOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<std::vector<int>>("Shape", "The shape of matrix to be randomized");
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

REGISTER_OP(random_op,
            paddle::operators::RandomOp,
            paddle::operators::RandomOpMaker);

typedef paddle::operators::RandomOpKernel<paddle::platform::CPUPlace, float>
    RandomOpKernel_CPU_float;
REGISTER_OP_CPU_KERNEL(random_op, RandomOpKernel_CPU_float);
