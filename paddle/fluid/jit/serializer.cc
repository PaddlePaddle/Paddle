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

#include "paddle/fluid/jit/serializer.h"

#include <set>

#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/jit/executor_function.h"
#include "paddle/fluid/jit/pe_function.h"
#include "paddle/fluid/jit/predictor_function.h"
#include "paddle/fluid/jit/serializer_utils.h"

DECLARE_string(function_type);

namespace paddle {
namespace jit {

Layer Deserializer::operator()(const std::string& path,
                               const phi::Place& place) {
  const auto& pdmodel_paths = utils::PdmodelFilePaths(path);
  // set is ordered
  std::set<std::string> param_names_set;
  std::vector<std::shared_ptr<FunctionInfo>> infos;
  Name2VariableMap params_dict;
  for (auto& it : pdmodel_paths) {
    auto& func_name = it.first;
    auto program_desc = LoadProgram(it.second);

    // TODO(dev): load int/float attrs
    std::vector<std::string> persist_var_names;
    auto all_var_desc = program_desc.Block(0).AllVars();
    for (auto* desc_ptr : all_var_desc) {
      if (utils::IsPersistable(desc_ptr)) {
        persist_var_names.emplace_back(desc_ptr->Name());
      }
    }

    param_names_set.insert(persist_var_names.begin(), persist_var_names.end());
    infos.emplace_back(std::make_shared<FunctionInfo>(
        func_name, persist_var_names, program_desc));
  }

  ReadTensorData(path + PDPARAMS_SUFFIX, param_names_set, place, &params_dict);
  // ReadAttributeData();

  Layer layer = Layer(infos, params_dict, place);

  for (auto& info : infos) {
    if (FLAGS_function_type == "Executor") {
      VLOG(3) << "Add function type: ExecutorFunction.";
      layer.SetFunction(
          info->FunctionName(),
          utils::MakeFunction<ExecutorFunction>(info, params_dict, place));
    } else if (FLAGS_function_type == "PE") {
      VLOG(3) << "Add function type: PEFunction.";
      layer.SetFunction(
          info->FunctionName(),
          utils::MakeFunction<PEFunction>(info, params_dict, place));
    } else if (FLAGS_function_type == "Predictor") {
      VLOG(3) << "Add function type: PredictorFunction.";
      layer.SetFunction(
          info->FunctionName(),
          utils::MakeFunction<PredictorFunction>(info, params_dict, place));
    } else {
      PD_THROW("Invalid JitLayer funciton type.");
    }
  }

  return layer;
}

void Deserializer::ReadTensorData(const std::string& file_name,
                                  const std::set<std::string>& var_name,
                                  const phi::Place& place,
                                  Name2VariableMap* params_dict) const {
  VLOG(3) << "ReadTensorData from: " << file_name;
  std::ifstream fin(file_name, std::ios::binary);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(place);
  for (auto it = var_name.begin(); it != var_name.end(); it++) {
    VLOG(3) << "load Tensor: " << *it;
    Variable v;
    // TODO(dev): Support framework::Vocab
    DenseTensor* dense_tesnor = v.GetMutable<DenseTensor>();
    framework::DeserializeFromStream(fin, dense_tesnor, dev_ctx);
    (*params_dict)[*it] = v;
  }
}

void Deserializer::ReadAttributeData(const std::string& file_path,
                                     Name2VariableMap* attrs_dict) const {}

framework::ProgramDesc Deserializer::LoadProgram(const std::string& file_name) {
  VLOG(3) << "LoadProgram from: " << file_name;
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  return framework::ProgramDesc(buffer);
}

Layer Load(const std::string& file_path, const phi::Place& place) {
  auto deserializer = Deserializer();
  return deserializer(file_path, place);
}

}  // namespace jit
}  // namespace paddle
