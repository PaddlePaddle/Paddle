// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/database/jsonfile_database.h"

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/auto_schedule/task/task_registry.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace auto_schedule {

// Return lowerd ir AST for example functions used in this test
std::vector<ir::LoweredFunc> LowerCompute(const std::vector<int>& shape,
                                          const Target& target) {
  CHECK(shape.size() == 2) << "shape should be 2";
  std::vector<Expr> domain;
  for (auto i = 0; i < shape.size(); ++i) {
    domain.emplace_back(shape[i]);
  }

  Placeholder<float> A("A", domain);
  ir::Tensor B, C;

  B = Compute(
      domain, [&A](Var i, Var j) { return A(i, j); }, "B");
  C = Compute(
      domain, [&B](Var i, Var j) { return B(i, j); }, "C");

  return cinn::lang::LowerVec(
      "test_func", CreateStages({A, B}), {A, B}, {}, {}, nullptr, target, true);
}

// Create a new IRSchedule with copied ir::LoweredFunc AST
ir::IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs,
                              const std::string& task_key) {
  std::vector<Expr> exprs;
  for (auto&& func : lowered_funcs) {
    exprs.emplace_back(ir::ir_utils::IRCopy(func->body));
  }
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  task_registry->Regist(task_key, ir::ModuleExpr(exprs));

  return ir::IRSchedule(ir::ModuleExpr(exprs));
}

class TestJSONFileDatabase : public ::testing::Test {
 public:
  TestJSONFileDatabase()
      : record_file_path("/tmp/test_record.json"),
        test_db(2, record_file_path, true) {}

  void SetUp() override { lowered_funcs = LowerCompute({32, 32}, target); }

  void TearDown() override {
    auto isFileExists = [](const std::string& file_path) -> bool {
      std::ifstream f(file_path.c_str());
      return f.good();
    };
    if (isFileExists(record_file_path)) {
      if (remove(record_file_path.c_str()) == 0) {
        LOG(INFO) << "Successfully deleted file: " << record_file_path;
      } else {
        LOG(INFO) << "failed to delete file: " << record_file_path;
      }
    } else {
      LOG(INFO) << "file: " << record_file_path << "does not exist.";
    }
  }

  std::string record_file_path;
  JSONFileDatabase test_db;
  std::vector<ir::LoweredFunc> lowered_funcs;
  Target target = common::DefaultHostTarget();
};

TEST_F(TestJSONFileDatabase, Serialize) {
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs, "test");
  auto fused = ir_sch.Fuse("B", {0, 1});
  VLOG(3) << "after Fuse, Expr: " << fused;

  TuningRecord record1("test", SearchState(std::move(ir_sch), 2.0), 1.0);
  std::string str = test_db.RecordToJSON(record1);
  VLOG(3) << "RecordToJSON: " << str;
  // Because the serialization of protobuf does not guarantee the order, we give
  // all possible results.
  std::string case1 =
      "{\"taskKey\":\"test\",\"executionCost\":1,\"predictedCost\":2,\"trace\":"
      "{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"loops_index\",\"dtype\":"
      "\"INTS\",\"ints\":[0,1]},{\"name\":\"block_"
      "name\",\"dtype\":\"STRING\",\"s\":\"B\"}]}]}}";
  std::string case2 =
      "{\"taskKey\":\"test\",\"executionCost\":1,\"predictedCost\":2,\"trace\":"
      "{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"block_name\",\"dtype\":"
      "\"STRING\",\"s\":\"B\"},{\"name\":\"loops_"
      "index\",\"dtype\":\"INTS\",\"ints\":[0,1]}]}]}}";
  EXPECT_EQ(true, str == case1 || str == case2);
}

TEST_F(TestJSONFileDatabase, SaveLoad) {
  ir::IRSchedule ir_sch1 = MakeIRSchedule(lowered_funcs, "k1");
  auto fused1 = ir_sch1.Fuse("B", {0, 1});
  ir::IRSchedule ir_sch2 = MakeIRSchedule(lowered_funcs, "k2");

  test_db.AddRecord(
      TuningRecord("k1", SearchState(std::move(ir_sch1), 1.5), 1.0));
  test_db.AddRecord(
      TuningRecord("k2", SearchState(std::move(ir_sch2), 3.5), 3.0));

  std::vector<std::string> strs = ReadLinesFromFile(record_file_path);
  ASSERT_EQ(strs.size(), 2);
  // Because the serialization of protobuf does not guarantee the order, we give
  // all possible results.
  std::string case1 =
      "{\"taskKey\":\"k1\",\"executionCost\":1,\"predictedCost\":1.5,\"trace\":"
      "{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"loops_index\",\"dtype\":"
      "\"INTS\",\"ints\":[0,1]},{\"name\":\"block_"
      "name\",\"dtype\":\"STRING\",\"s\":\"B\"}]}]}}";
  std::string case2 =
      "{\"taskKey\":\"k1\",\"executionCost\":1,\"predictedCost\":1.5,\"trace\":"
      "{\"steps\":[{\"type\":\"FuseWithName\","
      "\"outputs\":[\"e0\"],\"attrs\":[{\"name\":\"block_name\",\"dtype\":"
      "\"STRING\",\"s\":\"B\"},{\"name\":\"loops_"
      "index\",\"dtype\":\"INTS\",\"ints\":[0,1]}]}]}}";
  EXPECT_EQ(true, strs[0] == case1 || strs[0] == case2);
  EXPECT_EQ(strs[1],
            "{\"taskKey\":\"k2\",\"executionCost\":3,\"predictedCost\":3.5,"
            "\"trace\":{}}");
}

TEST_F(TestJSONFileDatabase, Basic) {
  test_db.AddRecord(TuningRecord(
      "k1", SearchState(MakeIRSchedule(lowered_funcs, "k1"), 1.0), 1.0));
  test_db.AddRecord(TuningRecord(
      "k2", SearchState(MakeIRSchedule(lowered_funcs, "k2"), 1.0), 2.0));
  test_db.AddRecord(TuningRecord(
      "k2", SearchState(MakeIRSchedule(lowered_funcs, "k2"), 1.0), 3.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 8.0), 3.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 7.0), 4.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 6.0), 5.0));
  test_db.AddRecord(TuningRecord(
      "k4", SearchState(MakeIRSchedule(lowered_funcs, "k4"), 1.0), 4.0));

  ASSERT_EQ(test_db.Size(), 6);
  auto records = test_db.LookUp("k3");
  // check the max number of stored candidates will
  // be restricted to capacity_per_task
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);
}

TEST_F(TestJSONFileDatabase, GetTopK) {
  test_db.AddRecord(TuningRecord(
      "k1", SearchState(MakeIRSchedule(lowered_funcs, "k1"), 1.0), 1.0));
  test_db.AddRecord(TuningRecord(
      "k2", SearchState(MakeIRSchedule(lowered_funcs, "k2"), 1.0), 2.0));
  test_db.AddRecord(TuningRecord(
      "k2", SearchState(MakeIRSchedule(lowered_funcs, "k2"), 1.0), 3.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 1.0), 3.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 1.0), 4.0));
  test_db.AddRecord(TuningRecord(
      "k3", SearchState(MakeIRSchedule(lowered_funcs, "k3"), 1.0), 5.0));
  test_db.AddRecord(TuningRecord(
      "k4", SearchState(MakeIRSchedule(lowered_funcs, "k4"), 2.0), 4.0));
  test_db.AddRecord(TuningRecord(
      "k4", SearchState(MakeIRSchedule(lowered_funcs, "k4"), 1.2), 2.0));
  test_db.AddRecord(TuningRecord(
      "k4", SearchState(MakeIRSchedule(lowered_funcs, "k4"), 1.0), 3.0));

  auto records = test_db.GetTopK("k4", 3);
  ASSERT_EQ(records.size(), 2);
  EXPECT_FLOAT_EQ(records[0].predicted_cost, 1.2);
  EXPECT_FLOAT_EQ(records[1].predicted_cost, 1.0);
}

TEST_F(TestJSONFileDatabase, Reload) {
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs, "k1");
  auto fused = ir_sch.Fuse("B", {0, 1});
  test_db.AddRecord(
      TuningRecord("k1", SearchState(std::move(ir_sch), 1.0), 1.0));
  test_db.AddRecord(TuningRecord(
      "k2", SearchState(MakeIRSchedule(lowered_funcs, "k2"), 1.0), 2.0));
  auto records = test_db.LookUp("k1");
  ASSERT_EQ(records.size(), 1);

  JSONFileDatabase new_db(2, record_file_path, false);
  ASSERT_EQ(new_db.Size(), 2);
  auto loaded_records = new_db.LookUp("k1");
  ASSERT_EQ(records.size(), loaded_records.size());
  EXPECT_EQ(records[0].task_key, loaded_records[0].task_key);
  EXPECT_EQ(records[0].execution_cost, loaded_records[0].execution_cost);
  EXPECT_EQ(records[0].predicted_cost, loaded_records[0].predicted_cost);

  // check the equality of trace info between original TuningRecord and the
  // loaded TuningRecord
  const auto& lhs_trace = records[0].trace;
  const auto& rhs_trace = loaded_records[0].trace;
  google::protobuf::util::MessageDifferencer dif;
  static const google::protobuf::Descriptor* descriptor =
      cinn::ir::proto::ScheduleDesc_Step::descriptor();
  dif.TreatAsSet(descriptor->FindFieldByName("attrs"));
  EXPECT_TRUE(dif.Compare(lhs_trace, rhs_trace));

  // check the equality of module expr between original TuningRecord
  // and the loaded TuningRecord by replaying with tracing ScheduleDesc
  ir::IRSchedule lhs_sch = MakeIRSchedule(lowered_funcs, "k1");
  ir::IRSchedule rhs_sch = MakeIRSchedule(lowered_funcs, "k1");
  ir::ScheduleDesc::ReplayWithProto(lhs_trace, &lhs_sch);
  ir::ScheduleDesc::ReplayWithProto(rhs_trace, &rhs_sch);
  auto lhs_exprs = lhs_sch.GetModule().GetExprs();
  auto rhs_exprs = rhs_sch.GetModule().GetExprs();

  ASSERT_EQ(lhs_exprs.size(), rhs_exprs.size());
  for (auto i = 0; i < lhs_exprs.size(); ++i) {
    std::string lhs = utils::GetStreamCnt(lhs_exprs.at(i));
    std::string rhs = utils::GetStreamCnt(rhs_exprs.at(i));
    size_t remove_prefix_len = 28;
    ASSERT_EQ(lhs.erase(0, remove_prefix_len), rhs.erase(0, remove_prefix_len));
  }
}

}  // namespace auto_schedule
}  // namespace cinn
