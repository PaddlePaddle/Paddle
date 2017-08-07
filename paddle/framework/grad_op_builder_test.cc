#include "paddle/framework/grad_op_builder.h"
#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP(add_two);

namespace paddle {
namespace framework {

class NOP : public OperatorBase {
 public:
  void InferShape(const Scope &scope) const override {}
  void Run(const Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {}
};

class MutiInOutOpMaker : public OpProtoAndCheckerMaker {
 public:
  MutiInOutOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("In1", "a single input");
    AddInput("In2_mult", "a multiple input").SetMultiple();
    AddInput("In3", "another single input");
    AddOutput("Out1", "a single output");
    AddOutput("Out2_mult", "a multiple output").SetMultiple();
    AddComment("test op with multiple inputs and outputs");
  }
};

class IOIgnoredOpMaker : public OpProtoAndCheckerMaker {
 public:
  IOIgnoredOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("In1", "a single input");
    AddInput("In2_mult", "a multiple input").SetMultiple().IgnoreGradient();
    AddInput("In3_mult", "another multiple input").SetMultiple();
    AddOutput("Out1_mult", "a multiple output").SetMultiple();
    AddOutput("Out2", "a single output").IgnoreGradient();
    AddComment("op with inputs and outputs ignored in gradient calculating");
  }
};

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;

TEST(GradOpBuilder, AddTwo) {
  std::shared_ptr<f::OperatorBase> add_op(
      f::OpRegistry::CreateOp("add_two", {"x", "y"}, {"out"}, {}));
  std::shared_ptr<f::OperatorBase> grad_add_op =
      f::OpRegistry::CreateGradOp(*add_op);
  EXPECT_EQ(static_cast<int>(grad_add_op->inputs_.size()), 4);
  EXPECT_EQ(static_cast<int>(grad_add_op->outputs_.size()), 2);
  EXPECT_EQ(grad_add_op->Input("X"), "x");
  EXPECT_EQ(grad_add_op->Input("Y"), "y");
  EXPECT_EQ(grad_add_op->Input("Out"), "out");
  EXPECT_EQ(grad_add_op->Input("Out@GRAD"), "out@GRAD");
  EXPECT_EQ(grad_add_op->Output("X@GRAD"), "x@GRAD");
  EXPECT_EQ(grad_add_op->Output("Y@GRAD"), "y@GRAD");
}

REGISTER_OP(mult_io, f::NOP, f::MutiInOutOpMaker);
REGISTER_GRADIENT_OP(mult_io, mult_io_grad, f::NOP);
REGISTER_OP(io_ignored, f::NOP, f::IOIgnoredOpMaker);
REGISTER_GRADIENT_OP(io_ignored, io_ignored_grad, f::NOP);

TEST(GradOpBuilder, MutiInOut) {
  f::AttributeMap attrs{{"input_format", std::vector<int>{0, 1, 4, 5}},
                        {"output_format", std::vector<int>{0, 1, 3}}};
  std::shared_ptr<f::OperatorBase> test_op(f::OpRegistry::CreateOp(
      "mult_io", {"in1", "in2_1", "in2_2", "in2_3", "in3"},
      {"out1", "out2_1", "out2_2"}, attrs));
  std::shared_ptr<f::OperatorBase> grad_test_op =
      f::OpRegistry::CreateGradOp(*test_op);

  ASSERT_EQ(grad_test_op->inputs_.size(), 5UL + 3UL + 3UL);
  EXPECT_EQ(grad_test_op->Input("In1"), "in1");
  EXPECT_EQ(grad_test_op->Inputs("In2_mult"),
            std::vector<std::string>({"in2_1", "in2_2", "in2_3"}));
  EXPECT_EQ(grad_test_op->Input("In3"), "in3");
  EXPECT_EQ(grad_test_op->Input("Out1"), "out1");
  EXPECT_EQ(grad_test_op->Inputs("Out2_mult"),
            std::vector<std::string>({"out2_1", "out2_2"}));
  EXPECT_EQ(grad_test_op->Input("Out1" + f::kGradVarSuffix),
            "out1" + f::kGradVarSuffix);
  EXPECT_EQ(grad_test_op->Inputs("Out2_mult" + f::kGradVarSuffix),
            std::vector<std::string>(
                {"out2_1" + f::kGradVarSuffix, "out2_2" + f::kGradVarSuffix}));

  ASSERT_EQ(grad_test_op->outputs_.size(), 5UL);
  EXPECT_EQ(grad_test_op->Output("In1" + f::kGradVarSuffix),
            "in1" + f::kGradVarSuffix);
  EXPECT_EQ(grad_test_op->Outputs("In2_mult" + f::kGradVarSuffix),
            std::vector<std::string>({"in2_1" + f::kGradVarSuffix,
                                      "in2_2" + f::kGradVarSuffix,
                                      "in2_3" + f::kGradVarSuffix}));
  EXPECT_EQ(grad_test_op->Output("In3" + f::kGradVarSuffix),
            "in3" + f::kGradVarSuffix);
}

TEST(GradOpBuilder, IOIgnoredInGradient) {
  f::AttributeMap attrs{{"input_format", std::vector<int>{0, 1, 3, 5}},
                        {"output_format", std::vector<int>{0, 2, 3}}};
  std::shared_ptr<f::OperatorBase> test_op(f::OpRegistry::CreateOp(
      "io_ignored", {"in1", "in2_1", "in2_2", "in3_1", "in3_2"},
      {"out1_1", "out1_2", "out2"}, attrs));
  std::shared_ptr<f::OperatorBase> grad_test_op =
      f::OpRegistry::CreateGradOp(*test_op);

  // 'In2' and 'Out2' are ignored in gradient calculating
  ASSERT_EQ(grad_test_op->inputs_.size(), 5UL + 3UL + 3UL);
  EXPECT_EQ(grad_test_op->Input("In1"), "in1");
  EXPECT_EQ(grad_test_op->Inputs("In2_mult"),
            std::vector<std::string>({f::kEmptyVarName, f::kEmptyVarName}));
  EXPECT_EQ(grad_test_op->Inputs("In3_mult"),
            std::vector<std::string>({"in3_1", "in3_2"}));
  EXPECT_EQ(grad_test_op->Inputs("Out1_mult"),
            std::vector<std::string>({"out1_1", "out1_2"}));
  EXPECT_EQ(grad_test_op->Input("Out2"), f::kEmptyVarName);
  EXPECT_EQ(grad_test_op->Inputs("Out1_mult" + f::kGradVarSuffix),
            std::vector<std::string>(
                {"out1_1" + f::kGradVarSuffix, "out1_2" + f::kGradVarSuffix}));
  EXPECT_EQ(grad_test_op->Input("Out2" + f::kGradVarSuffix),
            "out2" + f::kGradVarSuffix);

  ASSERT_EQ(grad_test_op->outputs_.size(), 5UL);
  EXPECT_EQ(grad_test_op->Output("In1" + f::kGradVarSuffix),
            "in1" + f::kGradVarSuffix);
  EXPECT_EQ(grad_test_op->Outputs("In2_mult" + f::kGradVarSuffix),
            std::vector<std::string>(
                {"in2_1" + f::kGradVarSuffix, "in2_2" + f::kGradVarSuffix}));
  EXPECT_EQ(grad_test_op->Outputs("In3_mult" + f::kGradVarSuffix),
            std::vector<std::string>(
                {"in3_1" + f::kGradVarSuffix, "in3_2" + f::kGradVarSuffix}));
}
