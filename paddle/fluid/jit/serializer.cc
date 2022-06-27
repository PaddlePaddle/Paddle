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

namespace paddle {
namespace jit {

Layer Deserializer::operator()(const std::string& path,
                               const phi::Place& place) {
  const auto& pdmodel_paths = GetPdmodelFilePaths(path);
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
      if (IsPersistable(desc_ptr)) {
        persist_var_names.emplace_back(desc_ptr->Name());
      }
    }

    param_names_set.insert(persist_var_names.begin(), persist_var_names.end());
    infos.emplace_back(std::make_shared<FunctionInfo>(
        func_name, persist_var_names, program_desc));
  }

  // Read from one pdiparams file, refine here
  ReadTensorData(path + PDPARAMS_SUFFIX, param_names_set, place, &params_dict);

  // ReadAttributeData();

  Layer layer = Layer(infos, params_dict, place);

  for (auto& info : infos) {
    VLOG(3) << "info->FunctionName(): " << info->FunctionName();
    layer.SetFunction(info->FunctionName(),
                      MakeFunction<ExecutorFunction>(info, params_dict, place));
  }

  return layer;
}

bool Deserializer::IsPersistable(framework::VarDesc* desc_ptr) {
  auto type = desc_ptr->GetType();
  if (type == framework::proto::VarType::FEED_MINIBATCH ||
      type == framework::proto::VarType::FETCH_LIST ||
      type == framework::proto::VarType::READER ||
      type == framework::proto::VarType::RAW) {
    return false;
  }
  return desc_ptr->Persistable();
}

bool Deserializer::StartWith(const std::string& str,
                             const std::string& prefix) {
  return str.compare(0, prefix.length(), prefix) == 0;
}

bool Deserializer::EndsWith(const std::string& str, const std::string& suffix) {
  if (str.length() < suffix.length()) {
    return false;
  }
  return str.compare(str.length() - suffix.length(), suffix.length(), suffix) ==
         0;
}

bool Deserializer::FileExists(const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  return file.good();
}

const std::vector<std::pair<std::string, std::string>>
Deserializer::GetPdmodelFilePaths(const std::string& path) {
  std::vector<std::pair<std::string, std::string>> pdmodel_paths;
  std::string layer_prefix = path.substr(path.find_last_of("/") + 1);
  std::string dir_path = path.substr(0, path.length() - layer_prefix.length());
  VLOG(3) << "layer_prefix:" << layer_prefix << "dir_path:" << dir_path;
  DIR* dir = opendir(dir_path.c_str());
  struct dirent* ptr;

  while ((ptr = readdir(dir)) != nullptr) {
    std::string file_name = ptr->d_name;

    if (StartWith(file_name, layer_prefix) &&
        EndsWith(file_name, PDMODEL_SUFFIX)) {
      std::string prefix = file_name.substr(
          0, file_name.length() - std::string(PDMODEL_SUFFIX).length());
      VLOG(3) << "prefix: " << prefix;
      std::string func_name = prefix.substr(prefix.find_first_of(".") + 1);
      VLOG(3) << "func_name:" << func_name << "path:" << dir_path + file_name;

      if (func_name == layer_prefix) {
        pdmodel_paths.emplace_back(
            std::make_pair("forward", dir_path + file_name));
      } else {
        pdmodel_paths.emplace_back(
            std::make_pair(func_name, dir_path + file_name));
      }
    }
  }
  closedir(dir);
  return pdmodel_paths;
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
  VLOG(3) << "LoadProgram " << file_name;
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
