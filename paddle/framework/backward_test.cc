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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace framework {

using DeviceContext = platform::DeviceContext;

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

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
namespace ops = paddle::operators;
using EnforceNotMet = paddle::platform::EnforceNotMet;
REGISTER_OPERATOR(rowwise_add, f::NOP, f::RowWiseAddOpMaker,
                  f::RowWiseAddGradMaker);
REGISTER_OPERATOR(rowwise_add_grad, f::NOP);
REGISTER_OP(mul, f::NOP, f::MulOpMaker, mul_grad, f::NOP);
REGISTER_OP(sigmoid, f::NOP, f::SigmoidOpMaker, sigmoid_grad, f::NOP);
REGISTER_OP_WITHOUT_GRADIENT(nograd, f::NOP, f::NoGradOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(fill_zeros_like, f::NOP, f::FillZeroOpMaker);
REGISTER_OP(sum, f::NOP, f::SumOpMaker, sum_grad, f::NOP);
REGISTER_OP_WITHOUT_GRADIENT(fc, f::FcOp, f::FcOpMaker);
REGISTER_OP(many_output_op, f::NOP, f::ManyOutputOpMaker, many_output_op_grad,
            f::NOP);

// TEST(Backward, simple_op_grad) {
//  auto fwd = f::OpRegistry::CreateOp(
//      "rowwise_add", {{"X", {"x"}}, {"b", {"b"}}}, {{"Out", {"out"}}}, {});
//  ASSERT_NE(fwd, nullptr);
//  auto gop = f::OpRegistry::CreateGradOp(*fwd);
//  ASSERT_EQ(1UL, gop->Inputs().size());
//  ASSERT_EQ("rowwise_add_grad", gop->Type());
//  ASSERT_EQ(f::GradVarName("x"), gop->Output(f::GradVarName("X")));
//  ASSERT_EQ(f::GradVarName("b"), gop->Output(f::GradVarName("b")));
//}

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
                + 2U /* internal variable number*/);

  EXPECT_EQ(grad_fc.Outputs(all).size(),
            2UL       /* input number of mul*/
                + 2UL /* input number of rowwise_add
                       */
                + 1UL /* input number of sigmod */);
  EXPECT_EQ(bwd_net->ops_[1]->Inputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[1]->Outputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[2]->Inputs(all).size(), 0UL);
  EXPECT_EQ(bwd_net->ops_[2]->Outputs(all).size(), 0UL);
}
