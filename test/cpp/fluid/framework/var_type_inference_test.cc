/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/var_type_inference.h"

#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class Scope;

class NOP : public OperatorBase {
 public:
  NOP(const std::string& type,
      const VariableNameMap& inputs,
      const VariableNameMap& outputs,
      const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope& scope, const phi::Place& place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    if (ctx->InputTypeAnyOf("X", proto::VarType::DENSE_TENSOR)) {
      default_var_type = proto::VarType::DENSE_TENSOR;
    }

    ctx->SetOutputType("Out", default_var_type);
  }
};
}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(fake_sum,
                  paddle::framework::NOP,
                  paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type,
                  paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {

class TestStaticGraphVarTypeInference : public StaticGraphVarTypeInference {
 public:
  void operator()(InferVarTypeContext* context) const override {}

  bool HasVar(InferVarTypeContext* ctx, const std::string& name) const {
    return StaticGraphVarTypeInference::HasVar(ctx, name);
  }

  const std::vector<std::string>& Input(InferVarTypeContext* ctx,
                                        const std::string& name) const {
    return StaticGraphVarTypeInference::Input(ctx, name);
  }

  const std::vector<std::string>& Output(InferVarTypeContext* ctx,
                                         const std::string& name) const {
    return StaticGraphVarTypeInference::Output(ctx, name);
  }

  proto::VarType::Type GetType(InferVarTypeContext* ctx,
                               const std::string& name) const {
    return StaticGraphVarTypeInference::GetType(ctx, name);
  }

  void SetType(InferVarTypeContext* ctx,
               const std::string& name,
               proto::VarType::Type type) const {
    StaticGraphVarTypeInference::SetType(ctx, name, type);
  }

  proto::VarType::Type GetDataType(InferVarTypeContext* ctx,
                                   const std::string& name) const {
    return StaticGraphVarTypeInference::GetDataType(ctx, name);
  }

  void SetDataType(InferVarTypeContext* ctx,
                   const std::string& name,
                   proto::VarType::Type type) const {
    StaticGraphVarTypeInference::SetDataType(ctx, name, type);
  }

  std::vector<proto::VarType::Type> GetDataTypes(
      InferVarTypeContext* ctx, const std::string& name) const {
    return StaticGraphVarTypeInference::GetDataTypes(ctx, name);
  }

  void SetDataTypes(
      InferVarTypeContext* ctx,
      const std::string& name,
      const std::vector<proto::VarType::Type>& multiple_data_type) {
    return StaticGraphVarTypeInference::SetDataTypes(
        ctx, name, multiple_data_type);
  }

  std::vector<int64_t> GetShape(InferVarTypeContext* ctx,
                                const std::string& name) const {
    return StaticGraphVarTypeInference::GetShape(ctx, name);
  }

  void SetShape(InferVarTypeContext* ctx,
                const std::string& name,
                const std::vector<int64_t>& dims) const {
    StaticGraphVarTypeInference::SetShape(ctx, name, dims);
  }

  int32_t GetLoDLevel(InferVarTypeContext* ctx, const std::string& name) const {
    return StaticGraphVarTypeInference::GetLoDLevel(ctx, name);
  }

  void SetLoDLevel(InferVarTypeContext* ctx,
                   const std::string& name,
                   int32_t lod_level) const {
    StaticGraphVarTypeInference::SetLoDLevel(ctx, name, lod_level);
  }
};

TEST(InferVarType, sum_op) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("fake_sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::DENSE_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::DENSE_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());
}

TEST(InferVarType, sum_op_without_infer_var_type) {
  ProgramDesc prog;
  auto* op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum_without_infer_var_type");
  op->SetInput("X", {"test2_a", "test2_b", "test2_c"});
  op->SetOutput("Out", {"test2_out"});

  prog.MutableBlock(0)->Var("test2_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType::DENSE_TENSOR,
            prog.MutableBlock(0)->Var("test2_out")->GetType());
}

TEST(InferVarType, multiple_api) {
  ProgramDesc prog;

  auto* block = prog.MutableBlock(0);
  auto* op = block->AppendOp();
  op->SetType("sum_without_infer_var_type");
  op->SetInput("X", {"test2_a", "test2_b"});
  op->SetOutput("Out", {"test2_a_out", "test2_b_out"});

  block->Var("test2_a")->SetType(proto::VarType::SELECTED_ROWS);
  block->Var("test2_b")->SetType(proto::VarType::SELECTED_ROWS);
  block->Var("test2_a_out");
  block->Var("test2_b_out");

  InferVarTypeContext ctx(op, block);

  ASSERT_TRUE(ctx.HasInput("X"));
  ASSERT_TRUE(ctx.HasOutput("Out"));

  ASSERT_EQ(2u, ctx.InputSize("X"));
  ASSERT_EQ("test2_a", ctx.InputVarName("X", 0));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS, ctx.GetInputType("X"));

  ASSERT_TRUE(ctx.InputTypeAllOf("X", proto::VarType::SELECTED_ROWS));
  ASSERT_FALSE(ctx.InputTypeAnyOf("X", proto::VarType::DENSE_TENSOR));

  ctx.SyncTypeAndDataType("X", "Out");

  ASSERT_EQ(proto::VarType::SELECTED_ROWS, ctx.GetOutputType("Out"));
  ASSERT_EQ(proto::VarType::DENSE_TENSOR, ctx.GetOutputType("Out", 1));

  ctx.SetOutputType("Out", proto::VarType::SELECTED_ROWS, ALL_ELEMENTS);
  ctx.SetOutputType("Out", proto::VarType::DENSE_TENSOR, 1);
  ASSERT_EQ(proto::VarType::SELECTED_ROWS, ctx.GetOutputType("Out"));
  ASSERT_EQ(proto::VarType::DENSE_TENSOR, ctx.GetOutputType("Out", 1));

  ASSERT_EQ(0, ctx.GetInputDataType("X"));

  ctx.SetOutputDataType("Out", proto::VarType::FP32, ALL_ELEMENTS);
  ctx.SetOutputDataType("Out", proto::VarType::INT8, 1);
  ASSERT_EQ(proto::VarType::FP32,
            prog.MutableBlock(0)->Var("test2_a_out")->GetDataType());
  ASSERT_EQ(proto::VarType::INT8,
            prog.MutableBlock(0)->Var("test2_b_out")->GetDataType());

  ASSERT_FALSE(ctx.IsDygraph());

  // test StaticGraphVarTypeInference
  TestStaticGraphVarTypeInference infer;
  ASSERT_TRUE(infer.HasVar(&ctx, "test2_a"));
  ASSERT_EQ(infer.Input(&ctx, "X").size(), infer.Output(&ctx, "Out").size());

  ASSERT_EQ(proto::VarType::FP32, infer.GetDataType(&ctx, "test2_a_out"));
  infer.SetDataType(&ctx, "test2_a_out", proto::VarType::FP64);
  ASSERT_EQ(proto::VarType::FP64, infer.GetDataType(&ctx, "test2_a_out"));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS, infer.GetType(&ctx, "test2_a_out"));
  infer.SetType(&ctx, "test2_a_out", proto::VarType::DENSE_TENSOR);
  ASSERT_EQ(proto::VarType::DENSE_TENSOR, infer.GetType(&ctx, "test2_a_out"));

  ASSERT_ANY_THROW(infer.GetDataTypes(&ctx, "test2_a_out"));
  ASSERT_ANY_THROW(infer.SetDataTypes(&ctx, "test2_a_out", {}));

  ASSERT_EQ(0u, infer.GetShape(&ctx, "test2_a_out").size());
  infer.SetShape(&ctx,
                 "test2_a_out",
                 {
                     1,
                     3,
                     3,
                 });
  ASSERT_EQ(3u, infer.GetShape(&ctx, "test2_a_out").size());

  ASSERT_EQ(0, infer.GetLoDLevel(&ctx, "test2_a_out"));
  infer.SetLoDLevel(&ctx, "test2_a_out", 2);
  ASSERT_EQ(2, infer.GetLoDLevel(&ctx, "test2_a_out"));
}

TEST(InferVarType, test_enforce_check) {
  InferVarTypeContext ctx(nullptr, nullptr);
  ASSERT_ANY_THROW(ctx.HasInput("X"));
  ASSERT_ANY_THROW(ctx.HasOutput("Out"));

  ASSERT_ANY_THROW(ctx.InputSize("X"));
  ASSERT_ANY_THROW(ctx.InputVarName("X"));

  ASSERT_ANY_THROW(ctx.InputTypeAnyOf("X", proto::VarType::DENSE_TENSOR));
  ASSERT_ANY_THROW(ctx.InputTypeAllOf("X", proto::VarType::DENSE_TENSOR));

  ASSERT_ANY_THROW(ctx.SyncTypeAndDataType("X", "Out"));

  ASSERT_ANY_THROW(ctx.SetOutputType("Out", proto::VarType::DENSE_TENSOR));
  ASSERT_ANY_THROW(ctx.GetInputType("X"));
  ASSERT_ANY_THROW(ctx.GetOutputType("Out"));

  ASSERT_ANY_THROW(ctx.GetInputDataType("X"));
  ASSERT_ANY_THROW(ctx.SetOutputDataType("Out", proto::VarType::DENSE_TENSOR));

  ASSERT_ANY_THROW(ctx.GetInputDataTypes("X"));
  ASSERT_ANY_THROW(ctx.SetOutputDataTypes("Out", {}));

  ASSERT_ANY_THROW(ctx.GetInputShape("X"));
  ASSERT_ANY_THROW(ctx.SetOutputShape("Out", {}));

  ASSERT_ANY_THROW(ctx.InsertVar("var", proto::VarType::DENSE_TENSOR));
}

}  // namespace framework
}  // namespace paddle
