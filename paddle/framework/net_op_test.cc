#include <gtest/gtest.h>
#include <paddle/framework/net.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

USE_OP_WITHOUT_KERNEL(test_operator);
USE_OP_WITHOUT_KERNEL(plainnet_operator);

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

  auto test_op = OpRegistry::CreateOp(test_op_desc);

  CPUDeviceContext cpu_device_context;
  auto scope = std::make_shared<Scope>();

  OperatorPtr op = paddle::framework::OpRegistry::CreateOp(net_op_desc);
  auto net_op = static_cast<PlainNet*>(op.get());

  net_op->AddOp(test_op_desc);
  net_op->AddOp(test_op);
  net_op->Run(scope, cpu_device_context);
}
