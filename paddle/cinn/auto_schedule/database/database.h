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

#pragma once
#include <unordered_map>

#include "paddle/cinn/auto_schedule/auto_schedule.pb.h"
#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/ir/schedule/schedule_desc.pb.h"

namespace cinn {
namespace auto_schedule {

// Record related data about tuning process of a measure candidate
struct TuningRecord {
  // the unique key to identify a task
  std::string task_key;
  // the predicted cost of CostModel
  float predicted_cost;  // unit: us
  // the ScheduleDesc of this tuning process
  ir::proto::ScheduleDesc trace;
  // the cost time of the candidate executed during measure
  double execution_cost;  // unit: us

  TuningRecord() = default;
  explicit TuningRecord(const proto::TuningRecord& record)
      : task_key(record.task_key()),
        predicted_cost(record.predicted_cost()),
        trace(record.trace()),
        execution_cost(record.execution_cost()) {}
  TuningRecord(const std::string& task_key,
               const SearchState& state,
               double execution_cost)
      : task_key(task_key),
        predicted_cost(state->predicted_cost),
        trace(state->ir_schedule.GetTraceDesc().ToProto()),
        execution_cost(execution_cost) {}

  // convert to proto object
  proto::TuningRecord ToProto() const;

  // a binary compare function that denotes when the left
  // will be sorted in the front of the right
  struct Compare {
    bool operator()(const TuningRecord& lhs, const TuningRecord& rhs) const;
  };
};

enum class DatabaseType : int { kMemory, kJSONFile };

struct DatabaseConfig {
  DatabaseType type = DatabaseType::kMemory;
  int capacity_per_task = 2;
  std::string record_file_path = "/tmp/tuning_record.json";
};

// A database supports insert or lookup historial tuning result with specified
// traits. It can be implemented with a concrete storage to save/load underlying
// data, such as memory, file, database server and so on, this base class can be
// regarded as one using memory as its underlying storage medium.
class Database {
 public:
  explicit Database(int capacity_per_task);
  ~Database() = default;

  // Create a Database with the specific config
  static std::unique_ptr<Database> Make(const DatabaseConfig& config);

  // add a record into the database
  bool AddRecord(const TuningRecord& record);
  // return all records whose task_keys are equal to the specified key
  std::vector<TuningRecord> LookUp(const std::string& task_key);
  // return the states of the top k in sorted candidates
  std::vector<TuningRecord> GetTopK(const std::string& task_key, int k);
  // return the total number of stored candidates
  size_t Size();
  // return the number of stored candidates with specified key
  size_t Count(const std::string& task_key);

 protected:
  // commit the newly added record into underlying storage
  virtual bool Commit(const TuningRecord& record) { return true; }
  // insert a newly added record into memory storage
  void Insert(const TuningRecord& record);

  // map task_key to its records
  std::unordered_map<std::string,
                     std::multiset<TuningRecord, TuningRecord::Compare>>
      key2record_;
  // the max number of candidates stored
  const int capacity_per_task_;
};

}  // namespace auto_schedule
}  // namespace cinn
