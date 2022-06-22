// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <set>
#include <string>

#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace jit {
static const char PDMODEL_SUFFIX[] = ".pdmodel";
static const char PDPARAMS_SUFFIX[] = ".pdiparams";

// Export Layer into local disk
class Serializer {
 public:
  void operator()(const Layer& layer, const std::string& file_dir);

  //  private:
  //   void WriteTensorData(const Layer& layer, const std::string& file_name)
  //   const;
  //   void WriteExtraInfo(const Layer& layer, const std::string& file_name)
  //   const;
  //   void WriteByteCode(const Layer& layer, const std::string& file_name)
  //   const;
};

class Deserializer {
 public:
  Layer operator()(const std::string& dir_path);

 private:
  bool IsPersistable(framework::VarDesc* desc_ptr);

  bool EndsWith(const std::string& str, const std::string& suffix);

  const std::vector<std::pair<std::string, std::string>>
  GetPdmodelFileNamePrefix(const std::string& path);

  void ReadTensorData(const std::string& file_name,
                      const std::set<std::string>& var_name,
                      const phi::Place& place,
                      VariableNameMap* params_dict) const;

  // void ReadExtraInfo(const std::string& file_name) const;
  // void ReadByteCode(const std::string& file_name) const;

  framework::ProgramDesc LoadProgram(const std::string& file_name);
};

void Export(const Layer& layer, const std::string& file_path);

Layer Load(const std::string& file_path);

}  // namespace jit
}  // namespace paddle
