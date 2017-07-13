#include <gtest/gtest.h>
#include <paddle/framework/net.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

namespace paddle {
namespace framework {
class OperatorTest : public OperatorBase {
 public:
  void Init() override { x = 1; }
  void InferShape(const std::shared_ptr<Scope>& scope) const override {}
  void Run(const std::shared_ptr<Scope>& scope,
           const platform::DeviceContext& dev_ctx) const override {
    float scale = GetAttr<float>("scale");
    ASSERT_NEAR(scale, 3.14, 1e-5);
    ASSERT_EQ(scope->GetVariable(inputs_[0]), nullptr);
    ASSERT_EQ(x, 1);
    std::cout << "this is test_operator" << std::endl;
  }

 public:
  float x = 0;
};

class OperatorTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OperatorTestProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddType("test_operator");
    AddComment("This is test op");
  }
};

REGISTER_OP(test_operator, OperatorTest, OperatorTestProtoAndCheckerMaker);
REGISTER_OP(plainnet_operator, PlainNet, PlainNetOpProtoAndCheckerMaker);

}  // namespace framework
}  // namespace paddle

TEST(OpKernel, all) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  // net op
  OpDesc net_op_desc;
  net_op_desc.set_type("plainnet_operator");

  // test op
  OpDesc test_op_desc;
  test_op_desc.set_type("test_operator");
  *test_op_desc.mutable_inputs()->Add() = "IN1";
  *test_op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = test_op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  CPUDeviceContext cpu_device_context;
  auto scope = std::make_shared<Scope>();

  OperatorBase* op = paddle::framework::OpRegistry::CreateOp(net_op_desc);
  auto net_op = static_cast<PlainNet*>(op);

  net_op->AddOp(test_op_desc);
  op->Run(scope, cpu_device_context);

  delete op;
}