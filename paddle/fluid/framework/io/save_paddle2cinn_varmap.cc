/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/io/save_paddle2cinn_varmap.h"
#include <fstream>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::framework {

void save_paddle2cinn_varmap(
    std::unordered_map<std::string, std::string> paddle2cinn_var_map,
    int64_t graph_compilation_key,
    std::string save_path) {
  std::stringstream ss;
  ss << "graph_compilation_key:" << std::to_string(graph_compilation_key)
     << "\n";
  for (const auto& kv : paddle2cinn_var_map) {
    ss << kv.first << ":" << kv.second << "\n";
  }
  std::string mapAsString = ss.str();

  // write string to save_path

  VLOG(6) << "paddle2cinn_varmap will be saved to " << save_path;
  MkDirRecursively(DirName(save_path).c_str());
  // set append mode to write all paddle var to cinn var map
  std::ofstream outfile(save_path, std::ios::app);
  PADDLE_ENFORCE_EQ(static_cast<bool>(outfile),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", save_path));
  outfile << mapAsString;
  outfile.close();
}

}  // namespace paddle::framework
