#include <gtest/gtest.h>
#define private public
#include <paddle/framework/op_registry.h>
USE_OP(add_two);
TEST(AddOp, GetOpProto) {
  auto& protos = paddle::framework::OpRegistry::protos();
  auto it = protos.find("add_two");
  ASSERT_NE(it, protos.end());
}