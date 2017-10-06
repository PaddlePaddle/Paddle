#include "paddle/framework/grad_op_builder.h"
#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(sum);

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

TEST(GradOpDescBuilder, MutiInOut) {
  f::OpDescBind *forw_op = new f::OpDescBind();
  forw_op->SetType("mult_io");
  forw_op->SetInput("In1", {"in1"});
  forw_op->SetInput("In2_mult", {"in2_1", "in2_2", "in2_3"});
  forw_op->SetInput("In3", {"in3"});
  forw_op->SetOutput("Out1", {"out1"});
  forw_op->SetOutput("Out2_mult", {"out2_1", "out2_2"});

  f::OpDescBind *grad_op = new f::OpDescBind();
  f::CompleteGradOpDesc(forw_op, grad_op);

  EXPECT_EQ(grad_op->Type(), "mult_io_grad");
  ASSERT_EQ(grad_op->InputNames().size(), 3UL + 2UL + 2UL);
  EXPECT_EQ(grad_op->Input("In1"), std::vector<std::string>({"in1"}));
  EXPECT_EQ(grad_op->Input("In2_mult"),
            std::vector<std::string>({"in2_1", "in2_2", "in2_3"}));
  EXPECT_EQ(grad_op->Input("In3"), std::vector<std::string>({"in3"}));
  EXPECT_EQ(grad_op->Input("Out1"), std::vector<std::string>({"out1"}));
  EXPECT_EQ(grad_op->Input("Out2_mult"),
            std::vector<std::string>({"out2_1", "out2_2"}));
  EXPECT_EQ(grad_op->Input(f::GradVarName("Out1")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op->Input(f::GradVarName("Out2_mult")),
            std::vector<std::string>(
                {f::GradVarName("out2_1"), f::GradVarName("out2_2")}));

  ASSERT_EQ(grad_op->OutputNames().size(), 3UL);
  EXPECT_EQ(grad_op->Output(f::GradVarName("In1")),
            std::vector<std::string>({f::GradVarName("in1")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("In2_mult")),
            std::vector<std::string>({f::GradVarName("in2_1"),
                                      f::GradVarName("in2_2"),
                                      f::GradVarName("in2_3")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("In3")),
            std::vector<std::string>({f::GradVarName("in3")}));
  delete forw_op;
  delete grad_op;
}

TEST(GradOpDescBuilder, IOIgnoredInGradient) {
  f::OpDescBind *forw_op = new f::OpDescBind();
  forw_op->SetType("io_ignored");
  forw_op->SetInput("In1", {"in1"});
  forw_op->SetInput("In2_mult", {"in2_1", "in2_2"});
  forw_op->SetInput("In3_mult", {"in3_1", "in3_2"});
  forw_op->SetOutput("Out1_mult", {"out1_1", "out1_2"});
  forw_op->SetOutput("Out2", {"out2"});

  f::OpDescBind *grad_op = new f::OpDescBind();
  f::CompleteGradOpDesc(forw_op, grad_op);

  EXPECT_EQ(grad_op->Type(), "io_ignored_grad");
  // 'In2' and 'Out2' are ignored in gradient calculating
  ASSERT_EQ(grad_op->InputNames().size(), 2UL + 1UL + 2UL);
  EXPECT_EQ(grad_op->Input("In1"), std::vector<std::string>({"in1"}));
  EXPECT_EQ(grad_op->Input("In3_mult"),
            std::vector<std::string>({"in3_1", "in3_2"}));
  EXPECT_EQ(grad_op->Input("Out1_mult"),
            std::vector<std::string>({"out1_1", "out1_2"}));
  EXPECT_EQ(grad_op->Input(f::GradVarName("Out1_mult")),
            std::vector<std::string>(
                {f::GradVarName("out1_1"), f::GradVarName("out1_2")}));
  EXPECT_EQ(grad_op->Input(f::GradVarName("Out2")),
            std::vector<std::string>({f::GradVarName("out2")}));

  ASSERT_EQ(grad_op->OutputNames().size(), 3UL);
  EXPECT_EQ(grad_op->Output(f::GradVarName("In1")),
            std::vector<std::string>({f::GradVarName("in1")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("In2_mult")),
            std::vector<std::string>(
                {f::GradVarName("in2_1"), f::GradVarName("in2_2")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("In3_mult")),
            std::vector<std::string>(
                {f::GradVarName("in3_1"), f::GradVarName("in3_2")}));
  delete forw_op;
  delete grad_op;
}
