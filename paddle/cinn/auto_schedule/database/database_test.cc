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

#include "paddle/cinn/auto_schedule/database/database.h"

#include <gtest/gtest.h>

#include <vector>

#include "paddle/cinn/auto_schedule/auto_schedule.pb.h"
#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TestDatabase : public ::testing::Test {
 public:
  TestDatabase() : test_db(2) {
    auto state = SearchState(ir::IRSchedule());
    test_db.AddRecord(TuningRecord("k1", state, 1.0));
    test_db.AddRecord(TuningRecord("k2", state, 2.0));
    test_db.AddRecord(TuningRecord("k2", state, 3.0));
    test_db.AddRecord(TuningRecord("k3", state, 3.0));
    test_db.AddRecord(TuningRecord("k3", state, 4.0));
    test_db.AddRecord(TuningRecord("k3", state, 5.0));
    test_db.AddRecord(TuningRecord("k4", state, 4.0));
  }

  void SetUp() override {}
  Database test_db;
};

TEST_F(TestDatabase, Basic) {
  ASSERT_EQ(test_db.Size(), 6);
  auto records = test_db.LookUp("k3");
  // check the max number of stored candidates will
  // be restricted to capacity_per_task
  ASSERT_EQ(test_db.Count("k3"), 2);
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0].execution_cost, 3.0);
  EXPECT_EQ(records[1].execution_cost, 4.0);
}

TEST_F(TestDatabase, GetTopK) {
  ASSERT_TRUE(test_db.GetTopK("k5", 2).empty());
  ASSERT_EQ(test_db.GetTopK("k4", 3).size(), 1);

  test_db.AddRecord(
      TuningRecord("k4", SearchState(ir::IRSchedule(), 1.2), 2.0));
  test_db.AddRecord(
      TuningRecord("k4", SearchState(ir::IRSchedule(), 1.0), 3.0));

  auto records = test_db.GetTopK("k4", 3);
  ASSERT_EQ(records.size(), 2);
  EXPECT_FLOAT_EQ(records[0].predicted_cost, 1.2);
  EXPECT_FLOAT_EQ(records[1].predicted_cost, 1.0);
}

}  // namespace auto_schedule
}  // namespace cinn
