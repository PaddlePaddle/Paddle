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
#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace jit {
using DenseTensor = phi::DenseTensor;
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
  Layer operator()(const std::string& dir_path) {
    const auto& file_name_prefixs = GetPdmodelFileNamePrefix(dir_path);
    std::vector<std::string> all_func_name;
    std::vector<framework::ProgramDesc> all_program_desc;
    std::vector<std::vector<std::string>> param_name_for_each_program;
    std::map<std::string, IValue> param_dict;
    for (auto& it : file_name_prefixs) {
      all_func_name.emplace_back(it.first);

      auto program = LoadProgram(dir_path + it.second + PDMODEL_SUFFIX);
      all_program_desc.emplace_back(program);

      std::vector<std::string> persistable_var_name;
      auto all_var_desc = program.Block(0).AllVars();
      for (auto* desc_ptr : all_var_desc) {
        if (IsPersistable(desc_ptr)) {
          persistable_var_name.emplace_back(desc_ptr->Name());
        }
      }
      // Sorting is required to correspond to the order of
      // parameters in the .pdparam file
      std::sort(persistable_var_name.begin(), persistable_var_name.end());
      param_name_for_each_program.emplace_back(persistable_var_name);

      auto param_for_program = ReadTensorData(
          dir_path + it.second + PDPARAMS_SUFFIX, persistable_var_name);
      // Now param is saved separately, gather all params
      param_dict.insert(param_for_program.begin(), param_for_program.end());
    }
    return Layer(all_func_name, all_program_desc, param_name_for_each_program,
                 param_dict);
  }

 private:
  bool IsPersistable(framework::VarDesc* desc_ptr) {
    auto type = desc_ptr->GetType();
    if (type == framework::proto::VarType::FEED_MINIBATCH ||
        type == framework::proto::VarType::FETCH_LIST ||
        type == framework::proto::VarType::READER ||
        type == framework::proto::VarType::RAW) {
      return false;
    }
    return desc_ptr->Persistable();
  }

  bool EndsWith(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
      return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(),
                       suffix) == 0;
  }

  const std::vector<std::pair<std::string, std::string>>
  GetPdmodelFileNamePrefix(const std::string& path) {
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

  std::map<std::string, IValue> ReadTensorData(
      const std::string& file_name,
      const std::vector<std::string>& var_name) const {
    VLOG(3) << "ReadTensorData from: " << file_name;
    std::ifstream fin(file_name, std::ios::binary);
    std::map<std::string, IValue> res;
    for (size_t i = 0; i < var_name.size(); ++i) {
      VLOG(3) << "load Tensor: " << var_name[i];
      // TODO(dev): support other tensor type
      Tensor t(std::make_shared<DenseTensor>());
      LoadTensorFromBuffer(fin, dynamic_cast<DenseTensor*>(t.impl().get()));
      t.set_name(var_name[i]);
      res[var_name[i]] = IValue(t);
    }
    return res;
  }

  void LoadTensorFromBuffer(std::istream& buffer, DenseTensor* tensor) const {
    {
      // the 1st field, unit32_t version for LoDTensor
      uint32_t version;
      buffer.read(reinterpret_cast<char*>(&version), sizeof(version));
      VLOG(3) << "version: " << version;
    }
    {
      // the 2st field, LoD information
      uint64_t lod_level;
      buffer.read(reinterpret_cast<char*>(&lod_level), sizeof(lod_level));
      auto& lod = *tensor->mutable_lod();
      // TODO(dev): support lod;
      lod.resize(lod_level);
      VLOG(3) << "lod_level: " << lod_level;
    }
    {
      // the 3st filed, Tensor
      {
        uint32_t version;
        buffer.read(reinterpret_cast<char*>(&version), sizeof(version));
        VLOG(3) << "tensor version: " << version;
      }
      framework::proto::VarType::TensorDesc desc;
      {
        int32_t size;
        buffer.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::unique_ptr<char[]> buf(new char[size]);
        buffer.read(reinterpret_cast<char*>(buf.get()), size);
        desc.ParseFromArray(buf.get(), size);
      }
      {
        // read tensor
        std::vector<int64_t> dims;
        dims.reserve(desc.dims().size());
        std::copy(desc.dims().begin(), desc.dims().end(),
                  std::back_inserter(dims));
        VLOG(3) << "dims: " << phi::make_ddim(dims);
        tensor->Resize(phi::make_ddim(dims));
        void* buf;
        size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
        buf = tensor->mutable_data(phi::CPUPlace());
        buffer.read(static_cast<char*>(buf), size);
      }
    }
  }

  // void ReadExtraInfo(const std::string& file_name) const;
  // void ReadByteCode(const std::string& file_name) const;

  framework::ProgramDesc LoadProgram(const std::string& file_name) {
    VLOG(3) << "LoadProgram " << file_name;
    std::ifstream fin(file_name, std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    std::string buffer(fin.tellg(), ' ');
    fin.seekg(0, std::ios::beg);
    fin.read(&buffer[0], buffer.size());
    fin.close();
    return framework::ProgramDesc(buffer);
  }
};

void Export(const Layer& layer, const std::string& file_path);

Layer Load(const std::string& file_path) {
  auto deserializer = Deserializer();
  return deserializer(file_path);
}

}  // namespace jit
}  // namespace paddle
