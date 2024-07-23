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
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace framework {

// Register test op
class FakeTestOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "").AsDuplicable();
    AddInput("Y", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class FakeTestOp : public OperatorBase {
 public:
  FakeTestOp(const std::string &type,
             const VariableNameMap &inputs,
             const VariableNameMap &outputs,
             const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope, const phi::Place &place) const override {
    // Fake RunImpl, for test only
    Variable *var = scope.FindVar("X");
    if (var != nullptr) {
      phi::DenseTensor *tensor = var->GetMutable<phi::DenseTensor>();
      tensor->mutable_data<float>(place);
    }
    int count = 0;
    while (count <= 1000) {
      ++count;
    }
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(fake_test_op,
                  paddle::framework::FakeTestOp,
                  paddle::framework::FakeTestOpMaker);

namespace paddle {
namespace framework {

ProgramDesc CreateTestProgram() {
  // create a ProgramDesc:
  //   Z = fake_test_op(X, Y)
  //   Out = fake_test_op(Z, W)
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

  auto *op0 = global_block->AppendOp();
  op0->SetType("fake_test_op");
  op0->SetInput("X", {x->Name()});
  op0->SetInput("Y", {y->Name()});

  auto *z = global_block->Var("Z");
  z->SetType(proto::VarType::LOD_TENSOR);
  op0->SetOutput("Out", {z->Name()});

  auto *w = global_block->Var("W");
  w->SetType(proto::VarType::LOD_TENSOR);
  w->SetLoDLevel(0);
  w->SetDataType(proto::VarType::FP32);
  w->SetShape({100, 10});

  auto *op1 = global_block->AppendOp();
  op1->SetType("fake_test_op");
  op1->SetInput("X", {z->Name()});
  op1->SetInput("Y", {w->Name()});

  auto *out = global_block->Var("Out");
  out->SetType(proto::VarType::LOD_TENSOR);
  op1->SetOutput("Out", {out->Name()});
  return program;
}

TEST(CostModelTest, TestProfileMeasure_EmptyProgram) {
  CostModel cost_model;
  ProgramDesc empty_program;
  CostData cost_data =
      cost_model.ProfileMeasure(empty_program, empty_program, "cpu", {"time"});
  EXPECT_EQ(cost_data.GetWholeTimeMs(), 0);
}

TEST(CostModelTest, TestProfileMeasure_Program) {
  CostModel cost_model;
  ProgramDesc program = CreateTestProgram();
  ProgramDesc empty_program;
  CostData cost_data =
      cost_model.ProfileMeasure(program, empty_program, "cpu", {"time"});
  double op0_time_ms = cost_data.GetOpTimeMs(0);
  double op1_time_ms = cost_data.GetOpTimeMs(1);
  EXPECT_GT(op0_time_ms, 0);
  EXPECT_GT(op1_time_ms, 0);
  EXPECT_GT(cost_data.GetWholeTimeMs(), op0_time_ms + op1_time_ms);
}

TEST(CostModelTest, TestProfileMeasure_UnsupportedDevice) {
  CostModel cost_model;
  ProgramDesc program = CreateTestProgram();
  ProgramDesc empty_program;

  EXPECT_THROW(cost_model.ProfileMeasure(
                   program, empty_program, "wrong_device", {"time"}),
               paddle::platform::EnforceNotMet);
}

TEST(CostDataTest, TestGetGraphProgram) {
  CostData cost_data;
  EXPECT_EQ(cost_data.GetGraph(), nullptr);
  EXPECT_EQ(cost_data.GetProgram(), nullptr);
}

TEST(CostDataTest, TestUninitialized) {
  CostData cost_data;
  EXPECT_EQ(cost_data.GetWholeMemoryBytes(), CostData::NOT_MEASURED);
  EXPECT_EQ(cost_data.GetWholeTimeMs(), CostData::NOT_MEASURED);
}

TEST(CostDataTest, TestEmptyProgram) {
  CostData cost_data;
  ProgramDesc empty_program("");
  EXPECT_EQ(cost_data.SetCostData(empty_program, {}), true);
  EXPECT_EQ(cost_data.GetWholeMemoryBytes(), 0);
  EXPECT_EQ(cost_data.GetWholeTimeMs(), 0);
}

TEST(CostDataTest, TestEmptyTimeEvent) {
  CostData cost_data;
  ProgramDesc program = CreateTestProgram();
  EXPECT_EQ(cost_data.SetCostData(program, {}), false);
  EXPECT_EQ(cost_data.GetWholeMemoryBytes(), CostData::NOT_MEASURED);
  EXPECT_EQ(cost_data.GetWholeTimeMs(), CostData::NOT_MEASURED);
}

TEST(CostDataTest, TestNoOpEvent) {
  CostData cost_data;
  ProgramDesc program = CreateTestProgram();
  std::vector<platform::Event> thread_events;
  thread_events.push_back(
      platform::Event(platform::EventType::kPushRange, "not exist name", 0));
  std::vector<std::vector<platform::Event>> time_events{thread_events};
  EXPECT_EQ(cost_data.SetCostData(program, time_events), false);
}

TEST(CostDataTest, TestNoOpPopEvent) {
  CostData cost_data;
  ProgramDesc program = CreateTestProgram();
  std::vector<platform::Event> thread_events;
  thread_events.push_back(
      platform::Event(platform::EventType::kPushRange, "fake_test_op", 0));
  std::vector<std::vector<platform::Event>> time_events{thread_events};
  EXPECT_EQ(cost_data.SetCostData(program, time_events), false);
}

TEST(CostDataTest, TestNoWholeEvent) {
  CostData cost_data;
  ProgramDesc program = CreateTestProgram();
  std::vector<platform::Event> thread_events;
  thread_events.push_back(
      platform::Event(platform::EventType::kPushRange, "fake_test_op", 0));
  thread_events.push_back(
      platform::Event(platform::EventType::kPopRange, "fake_test_op", 0));
  thread_events.push_back(
      platform::Event(platform::EventType::kPushRange, "fake_test_op", 0));
  thread_events.push_back(
      platform::Event(platform::EventType::kPopRange, "fake_test_op", 0));
  std::vector<std::vector<platform::Event>> time_events{thread_events};
  EXPECT_EQ(cost_data.SetCostData(program, time_events), false);
}

}  // namespace framework
}  // namespace paddle
