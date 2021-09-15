// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/cost_model.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

// Register test op
class FakeTestOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddInput("Y", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class FakeTestOp : public OperatorBase {
 public:
  FakeTestOp(const std::string &type, const VariableNameMap &inputs,
             const VariableNameMap &outputs, const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope,
               const platform::Place &place) const override {
    // Fake RunImpl, for test only
    int count = 0;
    while (count <= 1000) {
      ++count;
    }
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(fake_test_op, paddle::framework::FakeTestOp,
                  paddle::framework::FakeTestOpMaker);

namespace paddle {
namespace framework {

ProgramDesc CreateTestProgram() {
  // create a ProgramDesc: Out = fake_test_op(X, Y)
  ProgramDesc program;
  auto *global_block = program.MutableBlock(0);

  auto *x = global_block->Var("X");
  x->SetType(proto::VarType::LOD_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto *y = global_block->Var("Y");
  y->SetType(proto::VarType::LOD_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});

  auto *op = global_block->AppendOp();
  op->SetType("fake_test_op");
  op->SetInput("X", {x->Name()});
  op->SetInput("Y", {y->Name()});

  auto *out = global_block->Var("Out");
  out->SetType(proto::VarType::LOD_TENSOR);
  op->SetOutput("Out", {out->Name()});

  return program;
}

TEST(CostModelTest, TestProfileMeasure_Program) {
  CostModel cost_model;
  ProgramDesc program = CreateTestProgram();
  CostData cost_data = cost_model.ProfileMeasure(program, "cpu");
}

}  // namespace framework
}  // namespace paddle
