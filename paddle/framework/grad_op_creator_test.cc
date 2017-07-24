#include "paddle/framework/grad_op_creator.h"
#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(add_two);

namespace paddle {
namespace framework {

TEST(GradOpCreator, AddTwo) {
  OperatorPtr add_op(OpRegistry::CreateOp("add_two", {"x", "y"}, {"out"}, {}));
  OperatorPtr grad_add_op = OpRegistry::CreateGradOp(add_op);
  EXPECT_EQ(static_cast<int>(grad_add_op->inputs_.size()), 4);
  EXPECT_EQ(static_cast<int>(grad_add_op->outputs_.size()), 2);
  EXPECT_EQ(grad_add_op->Input("X"), "x");
  EXPECT_EQ(grad_add_op->Input("Y"), "y");
  EXPECT_EQ(grad_add_op->Input("Out"), "out");
  EXPECT_EQ(grad_add_op->Input("Out@GRAD"), "out@GRAD");
  EXPECT_EQ(grad_add_op->Output("X@GRAD"), "x@GRAD");
  EXPECT_EQ(grad_add_op->Output("Y@GRAD"), "y@GRAD");
}

}  // namespace framework
}  // namespace paddle