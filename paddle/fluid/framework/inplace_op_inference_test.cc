/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <iterator>
#include <string>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace framework {

class NOP : public OperatorBase {
 public:
  NOP(const std::string& type, const VariableNameMap& inputs,
      const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class SingleOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SingleGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("single_op_grad");
    op->SetInput("Out", OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return std::unique_ptr<OpDesc>(op);
  }
};

class SingleOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    ctx->HasInput("X");
    ctx->HasOutput("Out");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class SingleGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    ctx->HasInput(framework::GradVarName("Out"));
    ctx->HasOutput(framework::GradVarName("X"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }
};

class MultiOutOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddInput("Y", "").AsDuplicable();
    AddInput("Z", "").AsDuplicable();
    AddOutput("Out", "");
    AddOutput("YOut", "");
    AddOutput("ZOut", "");
    AddOutput("NotReuseOut", "");
    AddComment("");
  }
};

class MultiOutShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", "Out");
    ctx->ShareDim("Y", "YOut");
    ctx->ShareDim("Z", "ZOut");
  }
};

class MultiGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("multi_out_grad");
    op->SetInput("X", Input("X"));
    op->SetOutput(framework::GradVarName("Y"), OutputGrad("YOut"));
    op->SetOutput(framework::GradVarName("X"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Z"), OutputGrad("ZOut"));
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

class MultiOutGradShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Y"),
                      ctx->GetInputDim(framework::GradVarName("YOut")));
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
    ctx->SetOutputDim(framework::GradVarName("Z"),
                      ctx->GetInputDim(framework::GradVarName("ZOut")));
  }
};

class MultiOutInplaceInToOut : public framework::InplaceInToOut {
 public:
  using framework::InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const OpDesc& op_desc, BlockDesc* block) const override {
    return std::unordered_map<std::string, std::string>{
        {"X", "Out"}, {"Y", "YOut"}, {"Z", "ZOut"},
    };
  }
};

class MultiOutGradInplaceInToOut : public framework::InplaceInToOut {
 public:
  using framework::InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const OpDesc& op_desc, BlockDesc* block) const override {
    return std::unordered_map<std::string, std::string>{
        {framework::GradVarName("YOut"), framework::GradVarName("Y")},
        {framework::GradVarName("Out"), framework::GradVarName("X")},
        {framework::GradVarName("ZOut"), framework::GradVarName("Z")},
    };
  }
};

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
REGISTER_OPERATOR(single_op, f::NOP, f::SingleOpMaker, f::SingleGradOpMaker,
                  f::SingleOpInplaceInToOut, f::SingleOpShapeInference);
REGISTER_OPERATOR(single_op_grad, f::NOP, f::SingleOpInplaceInToOut,
                  f::SingleGradOpShapeInference);
REGISTER_OPERATOR(multi_out_op, f::NOP, f::MultiOutOpMaker, f::MultiGradOpMaker,
                  f::MultiOutInplaceInToOut, f::MultiOutShapeInference);
REGISTER_OPERATOR(multi_out_grad, f::NOP, f::MultiOutGradInplaceInToOut,
                  f::MultiOutGradShapeInference);

namespace paddle {
namespace framework {

TEST(InferInplace, SingleOpInplaceInToOut) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("single_op");
  op->SetInput("X", {"test2_a", "test2_b", "test2_c"});
  op->SetOutput("Out", {"test2_out"});

  prog.MutableBlock(0)->Var("test2_a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_a")->SetShape({32, 64, 128, 128});
  prog.MutableBlock(0)->Var("test2_b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_c")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_out");
  prog.MutableBlock(0)->Var("test2_out")->SetShape({32, 16, 128, 128});

  auto& infer_inplace = OpInfoMap::Instance().Get(op->Type()).infer_inplace_;
  auto in_to_outs = infer_inplace(*op, op->Block());
  EXPECT_EQ(in_to_outs.size(), 1ul);
  auto it = in_to_outs.begin();
  EXPECT_EQ(it->first, "test2_a");
  EXPECT_EQ(it->second, "test2_out");
}

TEST(InferInplace, SingleGradOpInplaceInToOut) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("single_op_grad");
  op->SetInput(GradVarName("Out"), {"test2_out"});
  op->SetOutput(GradVarName("X"), {"test2_a", "test2_b", "test2_c"});

  prog.MutableBlock(0)->Var("test2_a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_a")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("test2_b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_c")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("test2_out");
  prog.MutableBlock(0)->Var("test2_out")->SetShape({32, 16, 1024, 1024});

  auto& infer_inplace = OpInfoMap::Instance().Get(op->Type()).infer_inplace_;
  auto in_to_outs = infer_inplace(*op, op->Block());
  EXPECT_EQ(in_to_outs.size(), 1ul);
  auto it = in_to_outs.begin();
  EXPECT_EQ(it->first, "test2_out");
  EXPECT_EQ(it->second, "test2_a");
}

TEST(InferInplace, MultiOutInplaceInToOut) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("multi_out_op");
  op->SetInput("X", {"a0", "a1"});
  op->SetInput("Y", {"b0"});
  op->SetInput("Z", {"c0", "c1"});
  op->SetOutput("Out", {"o0"});
  op->SetOutput("YOut", {"y0"});
  op->SetOutput("ZOut", {"z0"});

  prog.MutableBlock(0)->Var("a0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c1")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("o0");
  prog.MutableBlock(0)->Var("y0");
  prog.MutableBlock(0)->Var("z0");
  prog.MutableBlock(0)->Var("a0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("b0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("c0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("o0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("y0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("z0")->SetShape({32, 16, 1024, 1024});

  auto& infer_inplace = OpInfoMap::Instance().Get(op->Type()).infer_inplace_;
  auto in_to_outs = infer_inplace(*op, op->Block());
  EXPECT_EQ(in_to_outs.size(), 3ul);
  std::unordered_map<std::string, std::string> expects = {
      {"a0", "o0"}, {"b0", "y0"}, {"c0", "z0"},
  };
  EXPECT_TRUE(expects == in_to_outs);
}

TEST(InferInplace, MultiGradInplaceInToOut) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("multi_out_grad");
  op->SetInput(GradVarName("Out"), {"o0"});
  op->SetInput(GradVarName("YOut"), {"y0"});
  op->SetInput(GradVarName("ZOut"), {"z0"});
  op->SetOutput(GradVarName("X"), {"a0", "a1"});
  op->SetOutput(GradVarName("Y"), {"b0"});
  op->SetOutput(GradVarName("Z"), {"c0", "c1"});

  prog.MutableBlock(0)->Var("a0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c0")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c1")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("o0");
  prog.MutableBlock(0)->Var("y0");
  prog.MutableBlock(0)->Var("z0");
  prog.MutableBlock(0)->Var("a0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("b0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("c0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("o0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("y0")->SetShape({32, 16, 1024, 1024});
  prog.MutableBlock(0)->Var("z0")->SetShape({32, 16, 1024, 1024});

  auto& infer_inplace = OpInfoMap::Instance().Get(op->Type()).infer_inplace_;
  auto in_to_outs = infer_inplace(*op, op->Block());

  EXPECT_EQ(in_to_outs.size(), 3ul);
  std::unordered_map<std::string, std::string> expects = {
      {"o0", "a0"}, {"y0", "b0"}, {"z0", "c0"},
  };
  EXPECT_TRUE(expects == in_to_outs);
}

}  // namespace framework
}  // namespace paddle
