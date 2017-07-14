#include "paddle/framework/op_registry.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
class CosineOp : public OperatorBase {
 public:
  void Run(const ScopePtr& scope,
           const platform::DeviceContext& dev_ctx) const override {}
  void InferShape(const ScopePtr& scope) const override {}
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CosineOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is cos op");
  }
};

class MyTestOp : public OperatorBase {
 public:
  void InferShape(const ScopePtr& scope) const override {}
  void Run(const ScopePtr& scope,
           const platform::DeviceContext& dev_ctx) const override {}

 public:
};

class MyTestOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  MyTestOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    auto my_checker = [](int i) {
      PADDLE_ENFORCE(i % 2 == 0, "'test_attr' must be even!");
    };
    AddAttr<int>("test_attr", "a simple test attribute")
        .AddCustomChecker(my_checker);
    AddComment("This is my_test op");
  }
};
}  // namespace framework
}  // namespace paddle

REGISTER_OP(cos_sim, paddle::framework::CosineOp,
            paddle::framework::CosineOpProtoAndCheckerMaker);
REGISTER_OP(my_test_op, paddle::framework::MyTestOp,
            paddle::framework::MyTestOpProtoAndCheckerMaker);

TEST(OpRegistry, CreateOp) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  float scale = 3.3;
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(scale);

  paddle::framework::OperatorPtr op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  auto scope = std::make_shared<paddle::framework::Scope>();
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  float scale_get = op->GetAttr<float>("scale");
  ASSERT_EQ(scale_get, scale);
}

TEST(OpRegistry, IllegalAttr) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(-2.0);

  bool caught = false;
  try {
    paddle::framework::OperatorPtr op __attribute__((unused)) =
        paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::framework::EnforceNotMet err) {
    caught = true;
    std::string msg = "larger_than check fail";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

TEST(OpRegistry, DefaultValue) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  ASSERT_TRUE(op_desc.IsInitialized());

  paddle::framework::OperatorPtr op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  auto scope = std::make_shared<paddle::framework::Scope>();
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  ASSERT_EQ(op->GetAttr<float>("scale"), 1.0);
}

TEST(OpRegistry, CustomChecker) {
  using namespace paddle::framework;

  paddle::framework::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  op_desc.add_inputs("ii");
  op_desc.add_outputs("oo");

  // attr 'test_attr' is not set
  bool caught = false;
  try {
    paddle::framework::OperatorPtr op __attribute__((unused)) =
        paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::framework::EnforceNotMet err) {
    caught = true;
    std::string msg = "Attribute 'test_attr' is required!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to an illegal value
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::framework::AttrType::INT);
  attr->set_i(3);
  caught = false;
  try {
    paddle::framework::OperatorPtr op __attribute__((unused)) =
        paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::framework::EnforceNotMet err) {
    caught = true;
    std::string msg = "'test_attr' must be even!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to a legal value
  op_desc.mutable_attrs()->Clear();
  attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::framework::AttrType::INT);
  attr->set_i(4);
  OperatorPtr op = paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::platform::CPUDeviceContext dev_ctx;
  auto scope = std::make_shared<paddle::framework::Scope>();
  op->Run(scope, dev_ctx);
  int test_attr = op->GetAttr<int>("test_attr");
  ASSERT_EQ(test_attr, 4);
}
