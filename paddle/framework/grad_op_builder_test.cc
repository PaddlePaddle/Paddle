#include "paddle/framework/grad_op_builder.h"
#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(add);

namespace paddle {
namespace framework {

class MutiInOutOpMaker : public OpProtoAndCheckerMaker {
 public:
  MutiInOutOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("In1", "a single input");
    AddInput("In2_mult", "a multiple input").AsDuplicable();
    AddInput("In3", "another single input");
    AddOutput("Out1", "a single output");
    AddOutput("Out2_mult", "a multiple output").AsDuplicable();
    AddComment("test op with multiple inputs and outputs");
  }
};

class IOIgnoredOpMaker : public OpProtoAndCheckerMaker {
 public:
  IOIgnoredOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("In1", "a single input");
    AddInput("In2_mult", "a multiple input").AsDuplicable().NotInGradient();
    AddInput("In3_mult", "another multiple input").AsDuplicable();
    AddOutput("Out1_mult", "a multiple output").AsDuplicable();
    AddOutput("Out2", "a single output").NotInGradient();
    AddComment("op with inputs and outputs ignored in gradient calculating");
  }
};

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;

TEST(GradOpBuilder, AddTwo) {
  std::shared_ptr<f::OperatorBase> add_op(f::OpRegistry::CreateOp(
      "add", {{"X", {"x"}}, {"Y", {"y"}}}, {{"Out", {"out"}}}, {}));
  std::shared_ptr<f::OperatorBase> grad_add_op =
      f::OpRegistry::CreateGradOp(*add_op);
  EXPECT_EQ(grad_add_op->Inputs().size(), 4UL);
  EXPECT_EQ(grad_add_op->Outputs().size(), 2UL);
  EXPECT_EQ(grad_add_op->Input("X"), "x");
  EXPECT_EQ(grad_add_op->Input("Y"), "y");
  EXPECT_EQ(grad_add_op->Input("Out"), "out");
  EXPECT_EQ(grad_add_op->Input(f::GradVarName("Out")), f::GradVarName("out"));
  EXPECT_EQ(grad_add_op->Output(f::GradVarName("X")), f::GradVarName("x"));
  EXPECT_EQ(grad_add_op->Output(f::GradVarName("Y")), f::GradVarName("y"));
}

REGISTER_OP(mult_io, f::NOP, f::MutiInOutOpMaker, mult_io_grad, f::NOP);
REGISTER_OP(io_ignored, f::NOP, f::IOIgnoredOpMaker, io_ignored_grad, f::NOP);

TEST(GradOpBuilder, MutiInOut) {
  std::shared_ptr<f::OperatorBase> test_op(f::OpRegistry::CreateOp(
      "mult_io", {{"In1", {"in1"}},
                  {"In2_mult", {"in2_1", "in2_2", "in2_3"}},
                  {"In3", {"in3"}}},
      {{"Out1", {"out1"}}, {"Out2_mult", {"out2_1", "out2_2"}}}, {}));
  std::shared_ptr<f::OperatorBase> grad_test_op =
      f::OpRegistry::CreateGradOp(*test_op);

  ASSERT_EQ(grad_test_op->Inputs().size(), 3UL + 2UL + 2UL);
  EXPECT_EQ(grad_test_op->Input("In1"), "in1");
  EXPECT_EQ(grad_test_op->Inputs("In2_mult"),
            std::vector<std::string>({"in2_1", "in2_2", "in2_3"}));
  EXPECT_EQ(grad_test_op->Input("In3"), "in3");
  EXPECT_EQ(grad_test_op->Input("Out1"), "out1");
  EXPECT_EQ(grad_test_op->Inputs("Out2_mult"),
            std::vector<std::string>({"out2_1", "out2_2"}));
  EXPECT_EQ(grad_test_op->Input(f::GradVarName("Out1")),
            f::GradVarName("out1"));
  EXPECT_EQ(grad_test_op->Inputs(f::GradVarName("Out2_mult")),
            std::vector<std::string>(
                {f::GradVarName("out2_1"), f::GradVarName("out2_2")}));

  ASSERT_EQ(grad_test_op->Outputs().size(), 3UL);
  EXPECT_EQ(grad_test_op->Output(f::GradVarName("In1")), f::GradVarName("in1"));
  EXPECT_EQ(grad_test_op->Outputs(f::GradVarName("In2_mult")),
            std::vector<std::string>({f::GradVarName("in2_1"),
                                      f::GradVarName("in2_2"),
                                      f::GradVarName("in2_3")}));
  EXPECT_EQ(grad_test_op->Output(f::GradVarName("In3")), f::GradVarName("in3"));
}

TEST(GradOpBuilder, IOIgnoredInGradient) {
  std::shared_ptr<f::OperatorBase> test_op(f::OpRegistry::CreateOp(
      "io_ignored", {{"In1", {"in1"}},
                     {"In2_mult", {"in2_1", "in2_2"}},
                     {"In3_mult", {"in3_1", "in3_2"}}},
      {{"Out1_mult", {"out1_1", "out1_2"}}, {"Out2", {"out2"}}}, {}));
  std::shared_ptr<f::OperatorBase> grad_test_op =
      f::OpRegistry::CreateGradOp(*test_op);

  // 'In2' and 'Out2' are ignored in gradient calculating
  ASSERT_EQ(grad_test_op->Inputs().size(), 2UL + 1UL + 2UL);
  EXPECT_EQ(grad_test_op->Input("In1"), "in1");
  EXPECT_EQ(grad_test_op->Inputs("In3_mult"),
            std::vector<std::string>({"in3_1", "in3_2"}));
  EXPECT_EQ(grad_test_op->Inputs("Out1_mult"),
            std::vector<std::string>({"out1_1", "out1_2"}));
  EXPECT_EQ(grad_test_op->Inputs(f::GradVarName("Out1_mult")),
            std::vector<std::string>(
                {f::GradVarName("out1_1"), f::GradVarName("out1_2")}));
  EXPECT_EQ(grad_test_op->Input(f::GradVarName("Out2")),
            f::GradVarName("out2"));

  ASSERT_EQ(grad_test_op->Outputs().size(), 3UL);
  EXPECT_EQ(grad_test_op->Output(f::GradVarName("In1")), f::GradVarName("in1"));
  EXPECT_EQ(grad_test_op->Outputs(f::GradVarName("In2_mult")),
            std::vector<std::string>(
                {f::GradVarName("in2_1"), f::GradVarName("in2_2")}));
  EXPECT_EQ(grad_test_op->Outputs(f::GradVarName("In3_mult")),
            std::vector<std::string>(
                {f::GradVarName("in3_1"), f::GradVarName("in3_2")}));
}
