// Copyright (c) 2023 Enflame Authors. All Rights Reserved.
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

#include <map>
#include <string>

namespace paddle {
namespace platform {
namespace gcu {
class GcuOptions {
 public:
  std::string GetOption(const std::string &key);

  void SetGraphOption(const std::string &key, const std::string &option);
  void SetGlobalOption(const std::string &key, const std::string &option);

  void ResetGraphOptions(std::map<std::string, std::string> options_map);
  void ResetGlobalOptions(std::map<std::string, std::string> options_map);

  std::map<std::string, std::string> GetAllGraphOptions() const;
  std::map<std::string, std::string> GetAllOptions() const;

  void ClearGraphOption(const std::string &key);
  void ClearGlobalOption(const std::string &key);
  void ClearAllOptions();

 private:
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> global_options_;
};  // class GcuOptions

GcuOptions &GetGcuOptions();

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
