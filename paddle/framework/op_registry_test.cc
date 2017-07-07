#include "paddle/framework/op_registry.h"
#include <gtest/gtest.h>

TEST(OpRegistry, CreateOp) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.3);

  paddle::framework::OpBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  std::string debug_str = op->Run();
  std::string str = "CosineOp runs! scale = " + std::to_string(3.3);
  ASSERT_EQ(str.size(), debug_str.size());
  for (size_t i = 0; i < debug_str.length(); ++i) {
    ASSERT_EQ(debug_str[i], str[i]);
  }
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
    paddle::framework::OpBase* op __attribute__((unused)) =
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

  paddle::framework::OpBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  std::string debug_str = op->Run();
  float default_value = 1.0;
  std::string str = "CosineOp runs! scale = " + std::to_string(default_value);
  ASSERT_EQ(str.size(), debug_str.size());
  for (size_t i = 0; i < debug_str.length(); ++i) {
    ASSERT_EQ(debug_str[i], str[i]);
  }
}

TEST(OpRegistry, CustomChecker) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  op_desc.add_inputs("ii");
  op_desc.add_outputs("oo");

  // attr 'test_attr' is not set
  bool caught = false;
  try {
    paddle::framework::OpBase* op __attribute__((unused)) =
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
    paddle::framework::OpBase* op __attribute__((unused)) =
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
  paddle::framework::OpBase* op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  std::string debug_str = op->Run();
  std::string str = "MyTestOp runs! test_attr = " + std::to_string(4);
  ASSERT_EQ(str.size(), debug_str.size());
  for (size_t i = 0; i < debug_str.length(); ++i) {
    ASSERT_EQ(debug_str[i], str[i]);
  }
}