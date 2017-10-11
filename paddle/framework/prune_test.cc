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

#include "paddle/framework/prune.h"

#include <gtest/gtest.h>
#include "paddle/framework/attribute.h"
#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"
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
REGISTER_OP(many_output_op, f::NOP, f::ManyOutputOpMaker, many_output_op_grad,
            f::NOP);
REGISTER_OP(mult_in_out, f::NOP, f::MultInOutOpMaker, mult_in_out_grad, f::NOP);

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           paddle::framework::BlockDescBind *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->NewVar(v);
      var->SetDataType(paddle::framework::DataType::FP32);
    }
  }

  // insert op
  auto op = block->AppendOp();
  op->SetType(type);
  for (auto &kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto &kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
}

f::ProgramDesc *GetNewProgramDesc() {
  auto *program_desc = new f::ProgramDesc();
  auto *root_block = program_desc->add_blocks();
  root_block->set_idx(0);
  root_block->set_parent_idx(-1);
  return program_desc;
}

TEST(Prune, one_operator) {
  f::ProgramDesc *program_desc = GetNewProgramDesc();
  f::ProgramDescBind &program = f::ProgramDescBind::Instance(program_desc);
  f::BlockDescBind *block = program.Block(0);

  AddOp("mul", {{"X", {"a"}}, {"Y", {"w1"}}}, {{"Out", {"b"}}}, {}, block);

  f::ProgramDesc *pdesc = program.Proto();
  f::ProgramDesc pruned;

  Prune(*pdesc, pruned, 0);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 0);

  pdesc->mutable_blocks(0)->mutable_ops(0)->set_is_target(true);
  Prune(*pdesc, pruned, 0);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 1);
}

TEST(Prune, simple_optimize) {}
