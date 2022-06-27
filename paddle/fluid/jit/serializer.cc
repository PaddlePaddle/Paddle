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
  // set is ordered
  std::set<std::string> param_names_set;
  std::vector<std::shared_ptr<FunctionInfo>> infos;
  Name2VariableMap params_dict;
  for (auto& it : file_name_prefixs) {
    auto& func_name = it.first;
    auto program_desc = LoadProgram(dir_path + it.second + PDMODEL_SUFFIX);

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

  auto default_place = imperative::GetCurrentTracer()->ExpectedPlace();
  // Read from one pdiparams file, refine here
  ReadTensorData(dir_path + "export.forward.pdiparams",
                 param_names_set,
                 default_place,
                 &params_dict);

  return Layer(infos, params_dict, default_place);
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

// process filename like `export.forward.pdmodel` and `export.infer.pdmodel`
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
