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

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include <fstream>

#include "paddle/cinn/auto_schedule/auto_schedule.pb.h"
#include "paddle/cinn/auto_schedule/task/task_registry.h"
#include "paddle/cinn/utils/multi_threading.h"

namespace cinn {
namespace auto_schedule {

// append a line to file
void AppendLineToFile(const std::string& file_path, const std::string& line) {
  std::ofstream os(file_path, std::ofstream::app);
  CHECK(os.good()) << "Cannot open the file to write: " << file_path;
  os << line << std::endl;
}

// read lines from a json file
std::vector<std::string> ReadLinesFromFile(const std::string& file_path,
                                           bool allow_new_file) {
  std::ifstream is(file_path);
  if (is.good()) {
    std::vector<std::string> json_strs;
    for (std::string str; std::getline(is, str);) {
      json_strs.push_back(str);
    }

    return json_strs;
  }
  CHECK(allow_new_file) << "File doesn't exist: " << file_path;
  std::ofstream os(file_path);
  CHECK(os.good()) << "Cannot create new file: " << file_path;
  return {};
}

JSONFileDatabase::JSONFileDatabase(int capacity_per_task,
                                   const std::string& record_file_path,
                                   bool allow_new_file)
    : Database(capacity_per_task), record_file_path_(record_file_path) {
  VLOG(3) << "Auto schedule will save/load tuning records on file:"
          << record_file_path;
  auto json_lines = ReadLinesFromFile(record_file_path_, allow_new_file);
  std::vector<cinn::auto_schedule::proto::TuningRecord> all_records_proto(
      json_lines.size());

  // convert JSON string to proto object
  auto worker_fn = [this, &json_lines, &all_records_proto](int index) {
    cinn::auto_schedule::proto::TuningRecord record_proto;
    auto status = google::protobuf::util::JsonStringToMessage(json_lines[index],
                                                              &record_proto);
    CHECK(status.ok()) << "Failed to parse JSON: " << json_lines[index];
    all_records_proto[index].Swap(&record_proto);
  };
  utils::parallel_run(
      worker_fn, utils::SequenceDispatcher(0, json_lines.size()), -1);

  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();

  for (const auto& record_proto : all_records_proto) {
    std::string task_key = record_proto.task_key();
    if (task_registry->Has(task_key)) {
      VLOG(4) << "Add a measured TuningRecord with task_key=" << task_key;
      Insert(TuningRecord(record_proto));
    }
  }
}

// convert a TuningRecord object to string in JSON format
std::string JSONFileDatabase::RecordToJSON(const TuningRecord& record) {
  proto::TuningRecord record_proto = record.ToProto();
  std::string json_string;
  auto status =
      google::protobuf::util::MessageToJsonString(record_proto, &json_string);
  CHECK(status.ok()) << "Failed to serialize record to JSON, task key = "
                     << record.task_key;
  VLOG(4) << "json_string = \n" << json_string;

  return json_string;
}

bool JSONFileDatabase::Commit(const TuningRecord& record) {
  std::string json_string = RecordToJSON(record);
  AppendLineToFile(record_file_path_, json_string);

  return true;
}

}  // namespace auto_schedule
}  // namespace cinn
