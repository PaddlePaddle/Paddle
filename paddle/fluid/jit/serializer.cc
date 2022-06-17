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

Layer Deserializer::operator()(const std::string& dir_path) {
  const auto& file_name_prefixs = GetPdmodelFileNamePrefix(dir_path);
  std::vector<std::string> func_names;
  std::vector<framework::ProgramDesc> program_descs;
  std::vector<std::vector<std::string>> param_names_for_each_program;
  // set is ordered
  std::set<std::string> param_names_set;
  VariableNameMap params_dict;
  for (auto& it : file_name_prefixs) {
    func_names.emplace_back(it.first);

    auto program = LoadProgram(dir_path + it.second + PDMODEL_SUFFIX);
    program_descs.emplace_back(program);

    // TODO(dev): load int/float params
    std::vector<std::string> persistable_var_names;
    auto all_var_desc = program.Block(0).AllVars();
    for (auto* desc_ptr : all_var_desc) {
      if (IsPersistable(desc_ptr)) {
        persistable_var_names.emplace_back(desc_ptr->Name());
      }
    }

    param_names_for_each_program.emplace_back(persistable_var_names);
    param_names_set.insert(persistable_var_names.begin(),
                           persistable_var_names.end());
  }

  auto default_place = imperative::GetCurrentTracer()->ExpectedPlace();
  // Read from one pdiparams file, refine here
  auto params_for_all_program = ReadTensorData(
      dir_path + "export.forward.pdiparams", param_names_set, default_place);
  params_dict.insert(params_for_all_program.begin(),
                     params_for_all_program.end());

  return Layer(func_names, program_descs, param_names_for_each_program,
               params_dict, default_place);
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

bool Deserializer::EndsWith(const std::string& str, const std::string& suffix) {
  if (str.length() < suffix.length()) {
    return false;
  }
  return str.compare(str.length() - suffix.length(), suffix.length(), suffix) ==
         0;
}

const std::vector<std::pair<std::string, std::string>>
Deserializer::GetPdmodelFileNamePrefix(const std::string& path) {
  std::vector<std::pair<std::string, std::string>> file_name_prefixs;
  DIR* dir = opendir(path.c_str());
  struct dirent* ptr;
  while ((ptr = readdir(dir)) != nullptr) {
    std::string file_name = ptr->d_name;
    if (EndsWith(file_name, PDMODEL_SUFFIX)) {
      std::string prefix = file_name.substr(
          0, file_name.length() - std::string(PDMODEL_SUFFIX).length());
      std::string func_name = prefix.substr(prefix.find_first_of(".") + 1);
      file_name_prefixs.emplace_back(std::make_pair(func_name, prefix));
    }
  }
  closedir(dir);
  return file_name_prefixs;
}

VariableNameMap Deserializer::ReadTensorData(
    const std::string& file_name, const std::set<std::string>& var_name,
    const phi::Place place) const {
  VLOG(3) << "ReadTensorData from: " << file_name;
  std::ifstream fin(file_name, std::ios::binary);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  // TODO(dev): Support other devices
  auto& dev_ctx = *pool.Get(place);
  VariableNameMap res;
  for (auto it = var_name.begin(); it != var_name.end(); it++) {
    VLOG(3) << "load Tensor: " << *it;
    Variable v;
    // TODO(dev): Support framework::Vocab
    DenseTensor* dense_tesnor = v.GetMutable<DenseTensor>();
    framework::DeserializeFromStream(fin, dense_tesnor, dev_ctx);
    res[*it] = v;
  }
  return res;
}

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

Layer Load(const std::string& file_path) {
  auto deserializer = Deserializer();
  return deserializer(file_path);
}

}  // namespace jit
}  // namespace paddle
