/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/backward.h"

#include <gtest/gtest.h>
#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/var_desc.h"
#include "paddle/operators/net_op.h"

USE_NO_KERNEL_OP(fill_constant);

namespace paddle {
namespace framework {

using DeviceContext = platform::DeviceContext;

class NoneOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {}
};

template <typename Place, typename T>
class NoneKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {}
};

class RowWiseAddOpMaker : public OpProtoAndCheckerMaker {
 public:
  RowWiseAddOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input X of Add");
    AddInput("b", "Bias of Add");
    AddOutput("Out", "Out of Add");
    AddComment("Add Op");
  }
};

class RowWiseAddGradMaker : public SingleGradOpDescMaker {
 public:
  using SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<OpDescBind> Apply() const override {
    auto grad_op = new OpDescBind();
    grad_op->SetInput(GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(GradVarName("X"), InputGrad("X"));
    grad_op->SetOutput(GradVarName("b"), InputGrad("b"));
    grad_op->SetType("rowwise_add_grad");
    return std::unique_ptr<OpDescBind>(grad_op);
  }
};

class MulOpMaker : public OpProtoAndCheckerMaker {
 public:
  MulOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "A");
    AddInput("Y", "B");
    AddOutput("Out", "Out");
    AddAttr<int>("x_num_col_dims", "").SetDefault(1).EqualGreaterThan(1);
    AddAttr<int>("y_num_col_dims", "").SetDefault(1).EqualGreaterThan(1);
    AddComment("Mul");
  }
};

class SigmoidOpMaker : public OpProtoAndCheckerMaker {
 public:
  SigmoidOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "X");
    AddOutput("Out", "Y");
    AddComment("Sigmoid");
  }
};

class NoGradOpMaker : public OpProtoAndCheckerMaker {
 public:
  NoGradOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "X input");
    AddOutput("Out", "Y output");
    AddComment("NoGradOp, same input output. no Grad");
  }
};

class FcOp : public operators::NetOp {
 public:
  FcOp(const std::string &type, const VariableNameMap &inputs,
       const VariableNameMap &outputs, const AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    AppendOp(OpRegistry::CreateOp("mul",
                                  {{"X", {Input("X")}}, {"Y", {Input("W")}}},
                                  {{"Out", {Output("mul_result")}}}, {}));
    auto input_b = Inputs("b");
    std::string before_act = "mul_result";
    if (input_b.size() != 0) {
      AppendOp(OpRegistry::CreateOp(
          "rowwise_add", {{"X", {Output("mul_result")}}, {"b", {input_b[0]}}},
          {{"Out", {Output("add_result")}}}, {}));
      before_act = "add_result";
    } else {
      auto out_varname = Output("add_result");
      if (out_varname != kEmptyVarName) {
        this->Rename(out_varname, kEmptyVarName);
      }
    }

    AppendOp(OpRegistry::CreateOp("sigmoid", {{"X", {Output(before_act)}}},
                                  {{"Out", {Output("Out")}}}, {}));
    CompleteAddOp(false);
  }
};

class FcOpMaker : public OpProtoAndCheckerMaker {
 public:
  FcOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "x");
    AddInput("W", "w");
    AddInput("b", "b");
    AddOutput("mul_result", "").AsIntermediate();
    AddOutput("add_result", "").AsIntermediate();
    AddOutput("Out", "");
    AddComment("");
  }
};

class ManyOutputOpMaker : public OpProtoAndCheckerMaker {
 public:
  ManyOutputOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("x", "x");
    AddOutput("y", "y");
    AddOutput("z", "z");
    AddComment("");
  }
};

class FillZeroOpMaker : public OpProtoAndCheckerMaker {
 public:
  FillZeroOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "x");
    AddOutput("Y", "out");
    AddComment("");
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SumOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensors of sum operator.").AsDuplicable();
    AddOutput("Out", "the output tensor of sum operator.");
    AddComment("");
  }
};

class MultInOutOpMaker : public OpProtoAndCheckerMaker {
 public:
  MultInOutOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "x");
    AddInput("H", "h");
    AddOutput("Y", "y");
    AddOutput("Z", "z");
    AddComment("");
  }
};

class MinusGradOpDescMaker : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<OpDescBind>> operator()() const override {
    std::vector<std::unique_ptr<OpDescBind>> retv;
    auto x_g = InputGrad("X");
    if (!x_g.empty()) {
      auto *op_desc = new OpDescBind();
      op_desc->SetType("scale");
      op_desc->SetInput("X", OutputGrad("Out"));
      op_desc->SetOutput("Out", x_g);
      op_desc->SetAttr("scale", 1.0f);
      retv.emplace_back(op_desc);
    }

    auto y_g = InputGrad("Y");
    if (!y_g.empty()) {
      auto *op_desc = new OpDescBind();
      op_desc->SetType("scale");
      op_desc->SetInput("X", OutputGrad("Out"));
      op_desc->SetOutput("Out", y_g);
      op_desc->SetAttr("scale", -1.0f);
      retv.emplace_back(op_desc);
    }
    return retv;
  }
};

class MinusOpMaker : public OpProtoAndCheckerMaker {
 public:
  MinusOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Y", "");
    AddOutput("Out", "");
    AddComment("minus for unittest");
  }
};
}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
namespace ops = paddle::operators;
using EnforceNotMet = paddle::platform::EnforceNotMet;
// rowwise_add
REGISTER_OPERATOR(rowwise_add, f::NoneOp, f::RowWiseAddOpMaker,
                  f::RowWiseAddGradMaker);
REGISTER_OP_CPU_KERNEL(rowwise_add,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
REGISTER_OPERATOR(rowwise_add_grad, f::NoneOp);
REGISTER_OP_CPU_KERNEL(rowwise_add_grad,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
// mul
REGISTER_OP(mul, f::NoneOp, f::MulOpMaker, mul_grad, f::NoneOp);
REGISTER_OP_CPU_KERNEL(mul, f::NoneKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(mul_grad,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
// sigmoid
REGISTER_OP(sigmoid, f::NoneOp, f::SigmoidOpMaker, sigmoid_grad, f::NoneOp);
REGISTER_OP_CPU_KERNEL(sigmoid,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_WITHOUT_GRADIENT(nograd, f::NoneOp, f::NoGradOpMaker);
// fill_zeros_like
REGISTER_OP_WITHOUT_GRADIENT(fill_zeros_like, f::NoneOp, f::FillZeroOpMaker);
REGISTER_OP_CPU_KERNEL(fill_zeros_like,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
// sum
REGISTER_OP(sum, f::NoneOp, f::SumOpMaker, sum_grad, f::NoneOp);
REGISTER_OP_CPU_KERNEL(sum, f::NoneKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(sum_grad,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
// fc
REGISTER_OP_WITHOUT_GRADIENT(fc, f::FcOp, f::FcOpMaker);
// many_output_op
REGISTER_OP(many_output_op, f::NoneOp, f::ManyOutputOpMaker,
            many_output_op_grad, f::NoneOp);
// mult_in_out
REGISTER_OP(mult_in_out, f::NoneOp, f::MultInOutOpMaker, mult_in_out_grad,
            f::NoneOp);
REGISTER_OP_CPU_KERNEL(mult_in_out,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(mult_in_out_grad,
                       f::NoneKernel<paddle::platform::CPUPlace, float>);
// minus
REGISTER_OPERATOR(minus, f::NoneOp, f::MinusOpMaker, f::MinusGradOpDescMaker);
REGISTER_OP_CPU_KERNEL(minus, f::NoneKernel<paddle::platform::CPUPlace, float>);
// scale
REGISTER_OPERATOR(scale, f::NoneOp);
REGISTER_OP_CPU_KERNEL(scale, f::NoneKernel<paddle::platform::CPUPlace, float>);

TEST(Backward, simple_op_not_need_grad) {
  auto fwd = f::OpRegistry::CreateOp(
      "rowwise_add", {{"X", {"x"}}, {"b", {"b"}}}, {{"Out", {"out"}}}, {});
  ASSERT_NE(fwd, nullptr);
  auto gop = f::Backward(*fwd, {"x"});
  ASSERT_EQ(gop->Output(f::GradVarName("X")), f::kEmptyVarName);

  auto no_input_gop = f::Backward(*fwd, {"x", "b"});
  ASSERT_NE(no_input_gop, nullptr);
  ASSERT_TRUE(no_input_gop->IsNetOp());
  ASSERT_EQ(0UL, static_cast<ops::NetOp *>(no_input_gop.get())->ops_.size());
}

TEST(Backward, net_fc_backward_normal) {
  std::shared_ptr<f::OperatorBase> fwd =
      f::OpRegistry::CreateOp("fc", {{"X", {"x"}}, {"W", {"w"}}, {"b", {"b"}}},
                              {{"mul_result", {"mul_res"}},
                               {"add_result", {"add_re"}},
                               {"Out", {"out"}}},
                              {});
  ASSERT_NE(fwd, nullptr);
  std::shared_ptr<f::OperatorBase> gop = f::Backward(*fwd, {});
  ASSERT_TRUE(gop->IsNetOp());
  auto net = static_cast<ops::NetOp *>(gop.get());

  ASSERT_NO_THROW(net->DebugString());

  ASSERT_EQ(3UL, net->ops_.size());

  f::OperatorBase &d_sigmoid = *net->ops_[0];
  ASSERT_EQ("sigmoid_grad", d_sigmoid.Type());

  f::OperatorBase &d_add = *net->ops_[1];
  ASSERT_EQ("rowwise_add_grad", d_add.Type());

  f::OperatorBase &d_mul = *net->ops_[2];
  ASSERT_EQ("mul_grad", d_mul.Type());
}

TEST(Backward, net_fc_backward_not_have_b) {
  std::shared_ptr<f::OperatorBase> fwd =
      f::OpRegistry::CreateOp("fc", {{"X", {"x"}}, {"W", {"w"}}, {"b", {}}},
                              {{"mul_result", {"mul_res"}},
                               {"add_result", {"add_res"}},
                               {"Out", {"tmp"}}},
                              {});
  ASSERT_NE(fwd, nullptr);
  std::shared_ptr<f::OperatorBase> gop = f::Backward(*fwd, {});
  ASSERT_TRUE(gop->IsNetOp());
  auto net = static_cast<ops::NetOp *>(gop.get());

  ASSERT_NO_THROW(net->DebugString());

  ASSERT_EQ(2UL, net->ops_.size());

  f::OperatorBase &d_sigmoid = *net->ops_[0];
  ASSERT_EQ("sigmoid_grad", d_sigmoid.Type());

  f::OperatorBase &d_mul = *net->ops_[1];
  ASSERT_EQ("mul_grad", d_mul.Type());
}

TEST(Backward, net_input_of_network_not_need_grad) {
  ops::NetOp net;
  net.AppendOp(f::OpRegistry::CreateOp(
      "fc", {{"X", {"x"}}, {"W", {"W1"}}, {"b", {"b1"}}},
      {{"mul_result", {"mul_tmp_0"}},
       {"add_result", {"add_tmp_0"}},
       {"Out", {"hidden0"}}},
      {}));
  net.AppendOp(f::OpRegistry::CreateOp(
      "fc", {{"X", {"hidden0"}}, {"W", {"W2"}}, {"b", {"b2"}}},
      {{"mul_result", {"mul_tmp_1"}},
       {"add_result", {"add_tmp_1"}},
       {"Out", {"hidden1"}}},
      {}));
  net.CompleteAddOp();
  auto bwd = Backward(net, {"x"});  // x@GRAD is not need.
  ASSERT_TRUE(bwd->IsNetOp());
  auto bwd_net = static_cast<ops::NetOp *>(bwd.get());

  auto output_vars = bwd_net->OutputVars(true);
  std::unordered_set<std::string> all_outputs =
      std::unordered_set<std::string>(output_vars.begin(), output_vars.end());
  all_outputs.erase(f::kEmptyVarName);

  for (auto &out : {"W1", "b1", "hidden0", "W2", "b2"}) {
    ASSERT_NE(all_outputs.find(f::GradVarName(out)), all_outputs.end());
  }

  // Not Generated X
  ASSERT_EQ(all_outputs.find(f::GradVarName("X")), all_outputs.end());

  ASSERT_EQ(2UL, bwd_net->ops_.size());
  ASSERT_TRUE(bwd_net->ops_[1]->IsNetOp());
  auto first_fc_grad = static_cast<ops::NetOp *>(bwd_net->ops_[1].get());
  ASSERT_EQ(3UL, first_fc_grad->ops_.size());
  ASSERT_EQ(f::kEmptyVarName,
            first_fc_grad->ops_[2]->Output(f::GradVarName("X")));
}

TEST(Backward, net_shared_weight) {
  ops::NetOp net;
  net.AppendOp(f::OpRegistry::CreateOp("mul", {{"X", {"x"}}, {"Y", {"w"}}},
                                       {{"Out", {"out"}}}, {}));
  net.AppendOp(f::OpRegistry::CreateOp("mul", {{"X", {"out"}}, {"Y", {"w"}}},
                                       {{"Out", {"FinalOut"}}}, {}));
  net.CompleteAddOp();

  auto bwd = f::Backward(net, {});
  ASSERT_TRUE(bwd->IsNetOp());
  auto bwd_net = static_cast<ops::NetOp *>(bwd.get());
  ASSERT_EQ(3UL, bwd_net->ops_.size());
  ASSERT_EQ("sum", bwd_net->ops_[2]->Type());
}

TEST(Backward, op_all_input_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp(
      "rowwise_add", {{"X", {"x"}}, {"b", {"b"}}}, {{"Out", {"out"}}}, {});
  auto backward = f::Backward(*fwd, {"x", "b"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<ops::NetOp *>(backward.get());
  ASSERT_TRUE(net->ops_.empty());
}

TEST(Backward, op_all_output_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp(
      "rowwise_add", {{"X", {"x"}}, {"b", {"b"}}}, {{"Out", {"out"}}}, {});
  auto backward = f::Backward(*fwd, {"out"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<ops::NetOp *>(backward.get());
  ASSERT_TRUE(net->ops_.empty());
}

TEST(Backward, op_part_of_output_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("many_output_op", {{"x", {"X"}}},
                                     {{"y", {"Y"}}, {"z", {"Z"}}}, {});
  auto backward = f::Backward(*fwd, {"Z"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<ops::NetOp *>(backward.get());
  ASSERT_EQ(net->ops_.size(), 2UL);

  auto &fill_zero = *net->ops_[0];
  ASSERT_EQ("fill_zeros_like", fill_zero.Type());
  ASSERT_EQ(1UL, fill_zero.Inputs("X").size());
  ASSERT_EQ("Z", fill_zero.Input("X"));
  ASSERT_EQ(1UL, fill_zero.Outputs("Y").size());
  ASSERT_EQ(std::string("Z") + f::kZeroVarSuffix, fill_zero.Output("Y"));

  auto &d_many_out = *net->ops_[1];
  ASSERT_EQ("many_output_op_grad", d_many_out.Type());
  ASSERT_EQ(1UL + 2UL + 2UL, d_many_out.Inputs().size());  // I/O/OG
  ASSERT_EQ(std::string("Z") + f::kZeroVarSuffix,
            d_many_out.Input(f::GradVarName("z")));
  ASSERT_EQ(f::GradVarName("Y"), d_many_out.Input(f::GradVarName("y")));
  ASSERT_EQ(f::GradVarName("X"), d_many_out.Output(f::GradVarName("x")));
}

TEST(Backward, op_part_of_input_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("mul", {{"X", {"a"}}, {"Y", {"b"}}},
                                     {{"Out", {"out"}}}, {});
  auto backward = f::Backward(*fwd, {"a"});
  auto &grad_mul = *backward;
  ASSERT_EQ(grad_mul.Type(), "mul_grad");
  ASSERT_EQ(grad_mul.Inputs().size(), 2UL + 1UL + 1UL);
  ASSERT_EQ(grad_mul.Outputs().size(), 2UL);
  ASSERT_EQ(grad_mul.Output(f::GradVarName("X")), f::kEmptyVarName);
  ASSERT_EQ(grad_mul.Output(f::GradVarName("Y")), f::GradVarName("b"));
  ASSERT_EQ(grad_mul.Input(f::GradVarName("Out")), f::GradVarName("out"));
  ASSERT_EQ(grad_mul.Input("X"), "a");
  ASSERT_EQ(grad_mul.Input("Y"), "b");
  ASSERT_EQ(grad_mul.Input("Out"), "out");
}

TEST(Backward, linear_net_intermediate_variable_has_no_grad) {
  ops::NetOp net;
  net.AppendOp(f::OpRegistry::CreateOp(
      "fc", {{"X", {"x1"}}, {"W", {"w1"}}, {"b", {"b1"}}},
      {{"mul_result", {"mul_out1"}},
       {"add_result", {"add_out1"}},
       {"Out", {"out1"}}},
      {}));
  net.AppendOp(f::OpRegistry::CreateOp(
      "fc", {{"X", {"out1"}}, {"W", {"w2"}}, {"b", {"b2"}}},
      {{"mul_result", {"mul_out2"}},
       {"add_result", {"tmp_out2"}},
       {"Out", {"out2"}}},
      {}));
  net.AppendOp(f::OpRegistry::CreateOp(
      "fc", {{"X", {"out2"}}, {"W", {"w3"}}, {"b", {"b3"}}},
      {{"mul_result", {"mul_out3"}},
       {"add_result", {"tmp_out3"}},
       {"Out", {"out3"}}},
      {}));
  net.CompleteAddOp();

  auto backward = f::Backward(net, {"mul_out2", "tmp_out2", "out2"});
  ASSERT_TRUE(backward->IsNetOp());
  auto bwd_net = static_cast<ops::NetOp *>(backward.get());
  ASSERT_EQ(bwd_net->ops_.size(), 3UL);
  auto &grad_fc = *bwd_net->ops_[0];

  const char *all = paddle::operators::NetOp::kAll;
  EXPECT_EQ(grad_fc.Inputs(all).size(),
            2UL       /* external input number */
                + 1UL /* external output number*/
                + 1UL /* number of gradient of external output*/
                + 2UL /* internal variable number*/
            );
  EXPECT_EQ(grad_fc.Outputs(all).size(),
            2UL       /* input number of mul*/
                + 2UL /* input number of rowwise_add*/
                + 1UL /* input number of sigmod */
                - 1UL /* out2 is not needed*/);
  EXPECT_EQ(bwd_net->ops_[1]->Inputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[1]->Outputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[2]->Inputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[2]->Outputs(all).size(), 0UL);
}

TEST(Backward, simple_single_op) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  f::OpDescBind *op = block->AppendOp();
  op->SetType("rowwise_add");
  op->SetInput("X", {"x"});
  op->SetInput("b", {"b"});
  op->SetOutput("Out", {"out"});

  auto target = f::VarDescBind("out");
  auto var_to_grad = AppendBackward(program, target, {});

  ASSERT_EQ(block->AllOps().size(), 3UL);
  f::OpDescBind *fill_op = block->AllOps()[1];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op = block->AllOps()[2];
  EXPECT_EQ(grad_op->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("x")}));
  EXPECT_EQ(grad_op->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b")}));

  EXPECT_EQ(var_to_grad.size(), 3UL);
  EXPECT_EQ(var_to_grad.at("b"), f::GradVarInfo(f::GradVarName("b"), 0, 2));
  EXPECT_EQ(var_to_grad.at("x"), f::GradVarInfo(f::GradVarName("x"), 0, 2));

  EXPECT_TRUE(block->HasVar(f::GradVarName("b")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("x")));
}

TEST(Backward, default_attribute) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  f::OpDescBind *op = block->AppendOp();
  op->SetType("mul");
  op->SetInput("X", {"x"});
  op->SetInput("Y", {"y"});
  op->SetOutput("Out", {"out"});
  op->CheckAttrs();

  auto target = f::VarDescBind("out");
  AppendBackward(program, target, {});

  ASSERT_EQ(block->AllOps().size(), 3UL);
  EXPECT_EQ(boost::get<int>(op->GetAttr("x_num_col_dims")), 1);
  EXPECT_EQ(boost::get<int>(op->GetAttr("y_num_col_dims")), 1);

  f::OpDescBind *fill_op = block->AllOps()[1];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op = block->AllOps()[2];
  ASSERT_EQ(grad_op->Type(), "mul_grad");
  EXPECT_EQ(boost::get<int>(grad_op->GetAttr("x_num_col_dims")), 1);
  EXPECT_EQ(boost::get<int>(grad_op->GetAttr("y_num_col_dims")), 1);
}

TEST(Backward, simple_mult_op) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  f::OpDescBind *op1 = block->AppendOp();
  op1->SetType("rowwise_add");
  op1->SetInput("X", {"x1"});
  op1->SetInput("b", {"b1"});
  op1->SetOutput("Out", {"out1"});

  f::OpDescBind *op2 = block->AppendOp();
  op2->SetType("mul");
  op2->SetInput("X", {"out1"});
  op2->SetInput("Y", {"y2"});
  op2->SetOutput("Out", {"out2"});

  f::OpDescBind *op3 = block->AppendOp();
  op3->SetType("rowwise_add");
  op3->SetInput("X", {"out2"});
  op3->SetInput("b", {"b3"});
  op3->SetOutput("Out", {"out3"});

  auto target = f::VarDescBind("out3");
  size_t forward_len = block->AllOps().size();
  auto var_to_grad = AppendBackward(program, target, {});

  ASSERT_EQ(block->AllOps().size(), 6UL + 1);
  f::OpDescBind *fill_op = block->AllOps()[forward_len];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op1 = block->AllOps()[6];
  EXPECT_EQ(grad_op1->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op1->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op1->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op1->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("x1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b1")}));

  f::OpDescBind *grad_op2 = block->AllOps()[5];
  EXPECT_EQ(grad_op2->Type(), "mul_grad");
  ASSERT_EQ(grad_op2->InputNames().size(), 4UL);
  ASSERT_EQ(grad_op2->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op2->Input("X"), std::vector<std::string>({"out1"}));
  EXPECT_EQ(grad_op2->Input("Y"), std::vector<std::string>({"y2"}));
  EXPECT_EQ(grad_op2->Input("Out"), std::vector<std::string>({"out2"}));
  EXPECT_EQ(grad_op2->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out2")}));
  EXPECT_EQ(grad_op2->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op2->Output(f::GradVarName("Y")),
            std::vector<std::string>({f::GradVarName("y2")}));

  f::OpDescBind *grad_op3 = block->AllOps()[4];
  EXPECT_EQ(grad_op3->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op3->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op3->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op3->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out3")}));
  EXPECT_EQ(grad_op3->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("out2")}));
  EXPECT_EQ(grad_op3->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b3")}));

  EXPECT_EQ(var_to_grad.size(), 7UL);
  EXPECT_EQ(var_to_grad.at("x1"), f::GradVarInfo(f::GradVarName("x1"), 0, 6));
  EXPECT_EQ(var_to_grad.at("b1"), f::GradVarInfo(f::GradVarName("b1"), 0, 6));
  EXPECT_EQ(var_to_grad.at("out1"),
            f::GradVarInfo(f::GradVarName("out1"), 0, 5));
  EXPECT_EQ(var_to_grad.at("y2"), f::GradVarInfo(f::GradVarName("y2"), 0, 5));
  EXPECT_EQ(var_to_grad.at("out2"),
            f::GradVarInfo(f::GradVarName("out2"), 0, 4));
  EXPECT_EQ(var_to_grad.at("b3"), f::GradVarInfo(f::GradVarName("b3"), 0, 4));

  EXPECT_TRUE(block->HasVar(f::GradVarName("x1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("b1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("out1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("y2")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("out2")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("b3")));
}

TEST(Backward, intermedia_var_no_grad) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  f::OpDescBind *op1 = block->AppendOp();
  op1->SetType("rowwise_add");
  op1->SetInput("X", {"x1"});
  op1->SetInput("b", {"b1"});
  op1->SetOutput("Out", {"out1"});

  f::OpDescBind *op2 = block->AppendOp();
  op2->SetType("mul");
  op2->SetInput("X", {"x2"});
  op2->SetInput("Y", {"y2"});
  op2->SetOutput("Out", {"out2"});

  f::OpDescBind *op3 = block->AppendOp();
  op3->SetType("rowwise_add");
  op3->SetInput("X", {"out2"});
  op3->SetInput("b", {"b3"});
  op3->SetOutput("Out", {"out3"});

  f::OpDescBind *op4 = block->AppendOp();
  op4->SetType("mul");
  op4->SetInput("X", {"out1"});
  op4->SetInput("Y", {"out3"});
  op4->SetOutput("Out", {"out4"});

  auto target = f::VarDescBind("out4");
  size_t forward_len = block->AllOps().size();
  auto var_to_grad = AppendBackward(program, target, {"out3"});

  ASSERT_EQ(block->AllOps().size(), 7UL);
  f::OpDescBind *fill_op = block->AllOps()[forward_len];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op1 = block->AllOps()[6];
  EXPECT_EQ(grad_op1->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op1->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op1->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op1->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("x1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b1")}));

  f::OpDescBind *grad_op4 = block->AllOps()[5];
  EXPECT_EQ(grad_op4->Type(), "mul_grad");
  ASSERT_EQ(grad_op4->InputNames().size(), 4UL);
  ASSERT_EQ(grad_op4->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op4->Input("X"), std::vector<std::string>({"out1"}));
  EXPECT_EQ(grad_op4->Input("Y"), std::vector<std::string>({"out3"}));
  EXPECT_EQ(grad_op4->Input("Out"), std::vector<std::string>({"out4"}));
  EXPECT_EQ(grad_op4->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out4")}));
  EXPECT_EQ(grad_op4->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op4->Output(f::GradVarName("Y")), std::vector<std::string>());

  EXPECT_EQ(var_to_grad.size(), 4UL);
  EXPECT_EQ(var_to_grad.at("x1"), f::GradVarInfo(f::GradVarName("x1"), 0, 6));
  EXPECT_EQ(var_to_grad.at("b1"), f::GradVarInfo(f::GradVarName("b1"), 0, 6));
  EXPECT_EQ(var_to_grad.at("out1"),
            f::GradVarInfo(f::GradVarName("out1"), 0, 5));

  EXPECT_TRUE(block->HasVar(f::GradVarName("x1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("b1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("out1")));
}

TEST(Backward, var_no_grad) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  f::OpDescBind *op1 = block->AppendOp();
  op1->SetType("mult_in_out");
  op1->SetInput("X", {"x1"});
  op1->SetInput("H", {"h1"});
  op1->SetOutput("Y", {"y1"});
  op1->SetOutput("Z", {"z1"});

  f::OpDescBind *op2 = block->AppendOp();
  op2->SetType("mult_in_out");
  op2->SetInput("X", {"y1"});
  op2->SetInput("H", {"z1"});
  op2->SetOutput("Y", {"y2"});
  op2->SetOutput("Z", {"z2"});

  auto target = f::VarDescBind("z2");
  size_t forward_len = block->AllOps().size();
  auto var_to_grad = AppendBackward(program, target, {"z1"});

  ASSERT_EQ(block->AllOps().size(), 6UL);
  f::OpDescBind *fill_op = block->AllOps()[forward_len];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op2 = block->AllOps()[3];
  ASSERT_EQ(grad_op2->Type(), "mult_in_out_grad");
  ASSERT_EQ(grad_op2->InputNames().size(), 6UL);
  ASSERT_EQ(grad_op2->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op2->Input("X"), std::vector<std::string>({"y1"}));
  EXPECT_EQ(grad_op2->Input("H"), std::vector<std::string>({"z1"}));
  EXPECT_EQ(grad_op2->Input("Y"), std::vector<std::string>({"y2"}));
  EXPECT_EQ(grad_op2->Input("Z"), std::vector<std::string>({"z2"}));
  EXPECT_EQ(grad_op2->Input(f::GradVarName("Y")),
            std::vector<std::string>({f::GradVarName("y2")}));
  EXPECT_EQ(grad_op2->Input(f::GradVarName("Z")),
            std::vector<std::string>({f::GradVarName("z2")}));
  EXPECT_EQ(grad_op2->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("y1")}));
  EXPECT_EQ(grad_op2->Output(f::GradVarName("H")), std::vector<std::string>());

  f::OpDescBind *fill_zero_op = block->AllOps()[4];
  ASSERT_EQ(fill_zero_op->Type(), "fill_zeros_like");
  ASSERT_EQ(fill_zero_op->InputNames().size(), 1UL);
  ASSERT_EQ(fill_zero_op->OutputNames().size(), 1UL);
  EXPECT_EQ(fill_zero_op->Input("X"), std::vector<std::string>({"z1"}));
  EXPECT_EQ(fill_zero_op->Output("Y"),
            std::vector<std::string>({std::string("z1") + f::kZeroVarSuffix}));

  f::OpDescBind *grad_op1 = block->AllOps()[5];
  ASSERT_EQ(grad_op1->Type(), "mult_in_out_grad");
  ASSERT_EQ(grad_op1->InputNames().size(), 6UL);
  ASSERT_EQ(grad_op1->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op1->Input("X"), std::vector<std::string>({"x1"}));
  EXPECT_EQ(grad_op1->Input("H"), std::vector<std::string>({"h1"}));
  EXPECT_EQ(grad_op1->Input("Y"), std::vector<std::string>({"y1"}));
  EXPECT_EQ(grad_op1->Input("Z"), std::vector<std::string>({"z1"}));
  EXPECT_EQ(grad_op1->Input(f::GradVarName("Y")),
            std::vector<std::string>({f::GradVarName("y1")}));
  EXPECT_EQ(grad_op1->Input(f::GradVarName("Z")),
            std::vector<std::string>({std::string("z1") + f::kZeroVarSuffix}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("x1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("H")),
            std::vector<std::string>({f::GradVarName("h1")}));

  EXPECT_EQ(var_to_grad.size(), 4UL);
  EXPECT_EQ(var_to_grad.at("y1"), f::GradVarInfo(f::GradVarName("y1"), 0, 3));
  EXPECT_EQ(var_to_grad.at("x1"), f::GradVarInfo(f::GradVarName("x1"), 0, 5));
  EXPECT_EQ(var_to_grad.at("h1"), f::GradVarInfo(f::GradVarName("h1"), 0, 5));

  EXPECT_TRUE(block->HasVar(f::GradVarName("y1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("x1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("h1")));
}

TEST(Backward, shared_var) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  f::OpDescBind *op1 = block->AppendOp();
  op1->SetType("rowwise_add");
  op1->SetInput("X", {"x1"});
  op1->SetInput("b", {"b1"});
  op1->SetOutput("Out", {"out1"});

  f::OpDescBind *op2 = block->AppendOp();
  op2->SetType("mul");
  op2->SetInput("X", {"out1"});
  op2->SetInput("Y", {"y2"});
  op2->SetOutput("Out", {"out2"});

  f::OpDescBind *op3 = block->AppendOp();
  op3->SetType("rowwise_add");
  op3->SetInput("X", {"out1"});
  op3->SetInput("b", {"b3"});
  op3->SetOutput("Out", {"out3"});

  auto target = f::VarDescBind("out3");
  size_t forward_len = block->AllOps().size();
  auto var_to_grad = AppendBackward(program, target, {});

  ASSERT_EQ(block->AllOps().size(), 8UL);
  f::OpDescBind *fill_op = block->AllOps()[forward_len];
  EXPECT_EQ(fill_op->Type(), "fill_constant");

  f::OpDescBind *grad_op3 = block->AllOps()[4];
  ASSERT_EQ(grad_op3->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op3->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op3->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op3->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out3")}));
  EXPECT_EQ(grad_op3->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("out1") + "@RENAME@0"}));
  EXPECT_EQ(grad_op3->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b3")}));

  f::OpDescBind *grad_op4 = block->AllOps()[5];
  ASSERT_EQ(grad_op4->Type(), "mul_grad");
  ASSERT_EQ(grad_op4->InputNames().size(), 4UL);
  ASSERT_EQ(grad_op4->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op4->Input("X"), std::vector<std::string>({"out1"}));
  EXPECT_EQ(grad_op4->Input("Y"), std::vector<std::string>({"y2"}));
  EXPECT_EQ(grad_op4->Input("Out"), std::vector<std::string>({"out2"}));
  EXPECT_EQ(grad_op4->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out2")}));
  EXPECT_EQ(grad_op4->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("out1") + "@RENAME@1"}));
  EXPECT_EQ(grad_op4->Output(f::GradVarName("Y")),
            std::vector<std::string>({f::GradVarName("y2")}));

  f::OpDescBind *sum_op = block->AllOps()[6];
  ASSERT_EQ(sum_op->Type(), "sum");
  ASSERT_EQ(sum_op->InputNames().size(), 1UL);
  ASSERT_EQ(sum_op->OutputNames().size(), 1UL);
  EXPECT_EQ(sum_op->Input("X"),
            std::vector<std::string>({f::GradVarName("out1") + "@RENAME@0",
                                      f::GradVarName("out1") + "@RENAME@1"}));
  EXPECT_EQ(sum_op->Output("Out"),
            std::vector<std::string>({f::GradVarName("out1")}));

  f::OpDescBind *grad_op1 = block->AllOps()[7];
  ASSERT_EQ(grad_op1->Type(), "rowwise_add_grad");
  ASSERT_EQ(grad_op1->InputNames().size(), 1UL);
  ASSERT_EQ(grad_op1->OutputNames().size(), 2UL);
  EXPECT_EQ(grad_op1->Input(f::GradVarName("Out")),
            std::vector<std::string>({f::GradVarName("out1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("X")),
            std::vector<std::string>({f::GradVarName("x1")}));
  EXPECT_EQ(grad_op1->Output(f::GradVarName("b")),
            std::vector<std::string>({f::GradVarName("b1")}));

  EXPECT_EQ(var_to_grad.size(), 6UL);
  EXPECT_EQ(var_to_grad.at("b3"), f::GradVarInfo(f::GradVarName("b3"), 0, 4));
  EXPECT_EQ(var_to_grad.at("y2"), f::GradVarInfo(f::GradVarName("y2"), 0, 5));
  EXPECT_EQ(var_to_grad.at("out1"),
            f::GradVarInfo(f::GradVarName("out1"), 0, 6));
  EXPECT_EQ(var_to_grad.at("x1"), f::GradVarInfo(f::GradVarName("x1"), 0, 7));
  EXPECT_EQ(var_to_grad.at("b1"), f::GradVarInfo(f::GradVarName("b1"), 0, 7));

  EXPECT_TRUE(block->HasVar(f::GradVarName("b3")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("y2")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("out1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("x1")));
  EXPECT_TRUE(block->HasVar(f::GradVarName("b1")));
}

TEST(Backward, half_backward) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);
  auto *op1 = block->AppendOp();
  op1->SetType("minus");
  op1->SetInput("X", {"a"});
  op1->SetInput("Y", {"b"});
  op1->SetOutput("Out", {"out"});

  auto target = f::VarDescBind("out");
  size_t forward_len = block->AllOps().size();
  auto var_to_grad = AppendBackward(program, target, {"b"});
  f::OpDescBind *fill_op = block->AllOps()[forward_len];
  EXPECT_EQ(fill_op->Type(), "fill_constant");
  auto ops = block->AllOps();
  ASSERT_EQ(3UL, ops.size());

  EXPECT_EQ(var_to_grad.size(), 2UL);
  EXPECT_EQ(var_to_grad.at("a"),
            f::GradVarInfo(f::GradVarName("a"), 0, forward_len + 1));
}
