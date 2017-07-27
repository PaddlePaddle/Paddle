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
#include "paddle/framework/net.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class EmptyOp : public OperatorBase {
 public:
  void InferShape(const std::shared_ptr<Scope> &scope) const override {}
  void Run(const std::shared_ptr<Scope> &scope,
           const platform::DeviceContext &dev_ctx) const override {}
};

class RowWiseAddOpMaker : public OpProtoAndCheckerMaker {
 public:
  RowWiseAddOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input X of Add").IgnoreGradient();
    AddInput("b", "Bias of Add").IgnoreGradient();
    AddOutput("Out", "Out of Add").IgnoreGradient();
    AddComment("Add Op");
  }
};

class MulOpMaker : public OpProtoAndCheckerMaker {
 public:
  MulOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("A", "A");
    AddInput("B", "B");
    AddOutput("Out", "Out");
    AddComment("Mul");
  }
};

class SigmoidOpMaker : public OpProtoAndCheckerMaker {
 public:
  SigmoidOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "X");
    AddOutput("Y", "Y");
    AddComment("Sigmoid");
  }
};

class FcOp : public NetOp {
 public:
  void Init() override {
    AddOp(OpRegistry::CreateOp("mul", {Input("X"), Input("W")},
                               {Output("before_act")}, {}));
    auto b_name = Input("b");
    if (b_name != EMPTY_VAR_NAME()) {
      AddOp(OpRegistry::CreateOp("rowwise_add", {Output("before_act"), b_name},
                                 {Output("before_act")}, {}));
    }
    AddOp(OpRegistry::CreateOp("sigmoid", {Output("before_act")},
                               {Output("Out")}, {}));
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
    AddOutput("before_act", "before act").SetTemporary();
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
    AddInput("x", "x");
    AddOutput("out", "out");
    AddComment("");
  }
};
}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
using EnforceNotMet = paddle::platform::EnforceNotMet;
REGISTER_OP(rowwise_add, f::EmptyOp, f::RowWiseAddOpMaker);
REGISTER_GRADIENT_OP(rowwise_add, rowwise_add_grad, f::EmptyOp);
REGISTER_OP(mul, f::EmptyOp, f::MulOpMaker);
REGISTER_GRADIENT_OP(mul, mul_grad, f::EmptyOp);
REGISTER_OP(sigmoid, f::EmptyOp, f::SigmoidOpMaker);
REGISTER_GRADIENT_OP(sigmoid, sigmoid_grad, f::EmptyOp);
REGISTER_OP(fc, f::FcOp, f::FcOpMaker);
REGISTER_OP(many_output_op, f::EmptyOp, f::ManyOutputOpMaker);
REGISTER_GRADIENT_OP(many_output_op, many_output_op_grad, f::EmptyOp);
REGISTER_OP(fill_zeros_like, f::EmptyOp, f::FillZeroOpMaker);

TEST(Backward, simple_grad) {
  auto fwd = f::OpRegistry::CreateOp("rowwise_add", {"X", "b"}, {"Out"}, {});
  ASSERT_NE(fwd, nullptr);
  auto gop = f::OpRegistry::CreateGradOp(*fwd);
  ASSERT_EQ(1, gop->inputs_.size());
  ASSERT_EQ("Out" + f::OperatorBase::GRAD_VAR_SUFFIX(), gop->inputs_[0]);
  ASSERT_EQ("rowwise_add_grad", gop->type_);
  ASSERT_EQ("X" + f::OperatorBase::GRAD_VAR_SUFFIX(), gop->outputs_[0]);
  ASSERT_EQ("b" + f::OperatorBase::GRAD_VAR_SUFFIX(), gop->outputs_[1]);

  ASSERT_EQ("X" + f::OperatorBase::GRAD_VAR_SUFFIX(),
            gop->Output("X" + f::OperatorBase::GRAD_VAR_SUFFIX()));
  //  LOG(INFO) << gop->Output("X" + "@GRAD");
}

TEST(Backward, fc_backward_normal) {
  std::shared_ptr<f::OperatorBase> fwd =
      f::OpRegistry::CreateOp("fc", {"X", "w", "b"}, {"out"}, {});
  ASSERT_NE(fwd, nullptr);
  std::shared_ptr<f::OperatorBase> gop = f::Backward(*fwd, {});
  ASSERT_TRUE(gop->IsNetOp());
  auto net = static_cast<f::NetOp *>(gop.get());

  ASSERT_NO_THROW(net->DebugString());

  ASSERT_EQ(3UL, net->ops_.size());

  f::OperatorBase &d_sigmoid = *net->ops_[0];
  ASSERT_EQ("sigmoid_grad", d_sigmoid.type_);

  f::OperatorBase &d_add = *net->ops_[1];
  ASSERT_EQ("rowwise_add_grad", d_add.type_);

  f::OperatorBase &d_mul = *net->ops_[2];
  ASSERT_EQ("mul_grad", d_mul.type_);
}

TEST(Backward, fc_backward_not_have_b) {
  std::shared_ptr<f::OperatorBase> fwd = f::OpRegistry::CreateOp(
      "fc", {"X", "w", f::OperatorBase::EMPTY_VAR_NAME()}, {"out"}, {});
  ASSERT_NE(fwd, nullptr);
  std::shared_ptr<f::OperatorBase> gop = f::Backward(*fwd, {});
  ASSERT_TRUE(gop->IsNetOp());
  auto net = static_cast<f::NetOp *>(gop.get());

  ASSERT_NO_THROW(net->DebugString());

  ASSERT_EQ(2UL, net->ops_.size());

  f::OperatorBase &d_sigmoid = *net->ops_[0];
  ASSERT_EQ("sigmoid_grad", d_sigmoid.type_);

  f::OperatorBase &d_mul = *net->ops_[1];
  ASSERT_EQ("mul_grad", d_mul.type_);
}

TEST(Backward, input_layer_not_need_grad) {
  f::NetOp net;
  net.AddOp(f::OpRegistry::CreateOp("fc", {"X", "W1", "b1"}, {"hidden0"}, {}));
  net.AddOp(
      f::OpRegistry::CreateOp("fc", {"hidden0", "W2", "b2"}, {"hidden1"}, {}));

  auto bwd = Backward(net, {"X"});  // X@GRAD is not need.
  ASSERT_TRUE(bwd->IsNetOp());
  auto bwd_net = static_cast<f::NetOp *>(bwd.get());

  std::unordered_set<std::string> all_output = std::unordered_set<std::string>(
      bwd_net->outputs_.begin(), bwd_net->outputs_.end());
  all_output.erase(f::OperatorBase::EMPTY_VAR_NAME());

  for (auto &out : {"W1", "b1", "hidden0", "W2", "b2"}) {
    ASSERT_NE(all_output.find(out + f::OperatorBase::GRAD_VAR_SUFFIX()),
              all_output.end());
  }
}

TEST(Backward, not_for_network) {
  auto fwd =
      f::OpRegistry::CreateOp("fc", {"X", "W", "b"}, {"Out", "tmp_out"},
                              {{"temporary_index", std::vector<int>{1}}});
  ASSERT_THROW(f::OpRegistry::CreateGradOp(*fwd), EnforceNotMet);
}

TEST(Backward, all_input_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("rowwise_add", {"X", "b"}, {"Out"}, {});
  auto backward = f::Backward(*fwd, {"X", "b"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<f::NetOp *>(backward.get());
  ASSERT_TRUE(net->ops_.empty());
}

TEST(Backward, all_output_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("rowwise_add", {"X", "b"}, {"Out"}, {});
  auto backward = f::Backward(*fwd, {"Out"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<f::NetOp *>(backward.get());
  ASSERT_TRUE(net->ops_.empty());
}

TEST(Backward, part_of_output_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("many_output_op", {"X"}, {"Y", "Z"}, {});
  auto backward = f::Backward(*fwd, {"Z"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<f::NetOp *>(backward.get());
  ASSERT_EQ(net->ops_.size(), 2UL);

  auto &fill_zero = *net->ops_[0];
  ASSERT_EQ("fill_zeros_like", fill_zero.type_);
  ASSERT_EQ(1UL, fill_zero.inputs_.size());
  ASSERT_EQ("Z", fill_zero.inputs_[0]);
  ASSERT_EQ(1UL, fill_zero.outputs_.size());
  ASSERT_EQ("Z" + f::OperatorBase::ZERO_VAR_SUFFIX(), fill_zero.outputs_[0]);

  auto &d_many_out = *net->ops_[1];
  ASSERT_EQ("many_output_op_grad", d_many_out.type_);
  ASSERT_EQ(1UL + 2UL + 2UL, d_many_out.inputs_.size());  // I/O/OG
  ASSERT_EQ("Z" + f::OperatorBase::ZERO_VAR_SUFFIX(),
            d_many_out.Input("z" + f::OperatorBase::GRAD_VAR_SUFFIX()));
  ASSERT_EQ("Y" + f::OperatorBase::GRAD_VAR_SUFFIX(),
            d_many_out.Input("y" + f::OperatorBase::GRAD_VAR_SUFFIX()));
  ASSERT_EQ("X" + f::OperatorBase::GRAD_VAR_SUFFIX(),
            d_many_out.Output("x" + f::OperatorBase::GRAD_VAR_SUFFIX()));
}

TEST(Backward, part_of_input_are_not_need) {
  auto fwd = f::OpRegistry::CreateOp("mul", {"a", "b"}, {"out"}, {});
  auto backward = f::Backward(*fwd, {"a"});
  ASSERT_TRUE(backward->IsNetOp());
  auto net = static_cast<f::NetOp *>(backward.get());
  ASSERT_EQ(net->ops_.size(), 1UL);

  auto &grad_mul = *net->ops_[0];
  ASSERT_EQ(grad_mul.type_, "mul_grad");
  ASSERT_EQ(grad_mul.inputs_.size(), 2UL + 1UL + 1UL);
  ASSERT_EQ(grad_mul.outputs_.size(), 2UL);
  ASSERT_EQ(grad_mul.Output("A" + f::OperatorBase::GRAD_VAR_SUFFIX()),
            f::OperatorBase::EMPTY_VAR_NAME());
  ASSERT_EQ(grad_mul.Output("B" + f::OperatorBase::GRAD_VAR_SUFFIX()),
            "b" + f::OperatorBase::GRAD_VAR_SUFFIX());
  ASSERT_EQ(grad_mul.Input("Out" + f::OperatorBase::GRAD_VAR_SUFFIX()),
            "out" + f::OperatorBase::GRAD_VAR_SUFFIX());
}