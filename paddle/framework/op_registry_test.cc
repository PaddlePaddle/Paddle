#include "paddle/framework/op_registry.h"
#include <gtest/gtest.h>
#include "paddle/framework/operator.h"
#include "paddle/operators/demo_op.h"

using namespace paddle::framework;

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

  paddle::framework::OperatorBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  auto scope = std::make_shared<Scope>();
  auto dev_ctx = DeviceContext();
  op->Run(scope, &dev_ctx);
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
    paddle::framework::OperatorBase* op __attribute__((unused)) =
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

  paddle::framework::OperatorBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  auto scope = std::make_shared<Scope>();
  auto dev_ctx = DeviceContext();
  op->Run(scope, &dev_ctx);
  ASSERT_EQ(op->GetAttr<float>("scale"), 1.0);
}

TEST(OpRegistry, CustomChecker) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  op_desc.add_inputs("ii");
  op_desc.add_outputs("oo");

  // attr 'test_attr' is not set
  bool caught = false;
  try {
    paddle::framework::OperatorBase* op __attribute__((unused)) =
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
    paddle::framework::OperatorBase* op __attribute__((unused)) =
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
  paddle::framework::OperatorBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  auto dev_ctx = DeviceContext();
  auto scope = std::make_shared<Scope>();
  op->Run(scope, &dev_ctx);
  int test_attr = op->GetAttr<int>("test_attr");
  ASSERT_EQ(test_attr, 4);
}
