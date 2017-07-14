#include <glog/logging.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

namespace paddle {
namespace operators {

class OperatorTest : public framework::OperatorBase {
public:
  void Init() override { x = 1; }
  void InferShape(const framework::ScopePtr& scope) const override {}
  void Run(const framework::ScopePtr& scope,
           const platform::DeviceContext& dev_ctx) const override {
    float scale = GetAttr<float>("scale");
    std::cout << "this is " << Type() << std::endl
              << " scale=" << scale << std::endl;
    std::cout << DebugString() << std::endl;
  }

public:
  float x = 0;
};

class OperatorTestProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
public:
  OperatorTestProtoAndCheckerMaker(framework::OpProto* proto,
                                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is test op");
  }
};

class OpKernelTestProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
public:
  OpKernelTestProtoAndCheckerMaker(framework::OpProto* proto,
                                   framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is test op");
  }
};

class OpWithKernelTest : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor*>& inputs,
      const std::vector<framework::Tensor*>& outputs) const override {}
};

class CPUKernelTest : public framework::OpKernel {
public:
  void Compute(const framework::OpKernel::KernelContext& context) const {
    float scale = context.op_.GetAttr<float>("scale");
    std::cout << "this is cpu kernel, scale=" << scale << std::endl;
    std::cout << context.op_.DebugString() << std::endl;
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(test_operator,
            paddle::operators::OperatorTest,
            paddle::operators::OperatorTestProtoAndCheckerMaker);
REGISTER_OP(op_with_kernel,
            paddle::operators::OpWithKernelTest,
            paddle::operators::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(op_with_kernel, paddle::operators::CPUKernelTest);
