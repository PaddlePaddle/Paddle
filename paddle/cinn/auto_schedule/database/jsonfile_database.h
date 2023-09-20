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

#include "paddle/cinn/auto_schedule/database/database.h"

namespace cinn {
namespace auto_schedule {

// JSONFileDatabase is a database implemented by JSON file to save/load
// underlying data.
class JSONFileDatabase : public Database {
 public:
  /*!
   * \brief Build a JSONFileDatabase object from a json file.
   * \param capacity_per_task The max number of candidates stored.
   * \param record_file_path The path of the json file.
   * \param allow_new_file Whether to create new file when the given path is not
   * found.
   */
  JSONFileDatabase(int capacity_per_task,
                   const std::string& record_file_path,
                   bool allow_new_file);
  ~JSONFileDatabase() = default;

  // convert a TuningRecord object to string in JSON format
  std::string RecordToJSON(const TuningRecord& record);

 protected:
  // commit the newly added record into json file
  bool Commit(const TuningRecord& record) override;

  // the name of the json file to save tuning records.
  std::string record_file_path_;
};

// append a line to file
void AppendLineToFile(const std::string& file_path, const std::string& line);

// read lines from a json file
std::vector<std::string> ReadLinesFromFile(const std::string& file_path,
                                           bool allow_new_file = true);

}  // namespace auto_schedule
}  // namespace cinn
