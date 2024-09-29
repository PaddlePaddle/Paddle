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

#include "paddle/fluid/jit/serializer_utils.h"

#include <dirent.h>
#include <fstream>

#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/var_desc.h"

#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
COMMON_DECLARE_bool(enable_pir_api);
namespace paddle::jit::utils {

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

bool IsPersistable(pir::Value* value_ptr) {
  auto is_persistable =
      value_ptr->attribute<pir::BoolAttribute>(kAttrIsPersistable);
  if (is_persistable && is_persistable.data()) {
    return true;
  }
  return false;
}

bool StartsWith(const std::string& str, const std::string& prefix) {
  return str.compare(0, prefix.length(), prefix) == 0;
}

bool EndsWith(const std::string& str, const std::string& suffix) {
  if (str.length() < suffix.length()) {
    return false;
  }
  return str.compare(str.length() - suffix.length(), suffix.length(), suffix) ==
         0;
}

void ReplaceAll(std::string* str,
                const std::string& old_value,
                const std::string& new_value) {
  std::string::size_type pos = 0;
  while ((pos = str->find(old_value, pos)) != std::string::npos) {
    *str = str->replace(pos, old_value.length(), new_value);
    if (new_value.length() > 0) {
      pos += new_value.length();
    }
  }
}

bool FileExists(const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  return file.good();
}

const std::vector<std::pair<std::string, std::string>> ModelFilePaths(
    const std::string& path) {
  std::vector<std::pair<std::string, std::string>> model_paths;
  std::string format_path = path;
  ReplaceAll(&format_path, R"(\\)", "/");
  ReplaceAll(&format_path, R"(\)", "/");
  if (format_path.find('/') == std::string::npos) {
    format_path = "./" + format_path;
  }

  std::string layer_name =
      format_path.substr(format_path.find_last_of('/') + 1);
  std::string dir_path =
      format_path.substr(0, format_path.length() - layer_name.length());
  DIR* dir = opendir(dir_path.c_str());
  struct dirent* ptr = nullptr;

  while ((ptr = readdir(dir)) != nullptr) {
    std::string prefix = "";
    std::string file_name = ptr->d_name;
    if (FLAGS_enable_pir_api) {
      if (StartsWith(file_name, layer_name) &&
          EndsWith(file_name, JSON_SUFFIX)) {
        std::string prefix = file_name.substr(
            0, file_name.length() - std::string(JSON_SUFFIX).length());

        if (prefix == layer_name) {
          model_paths.emplace_back(
              std::make_pair("forward", dir_path + file_name));
        } else {
          std::string func_name = prefix.substr(layer_name.size() + 1);
          model_paths.emplace_back(
              std::make_pair(func_name, dir_path + file_name));
        }
        VLOG(3) << "func_name: " << model_paths.back().first
                << ", path:" << dir_path + file_name;
      }
    } else {
      if (StartsWith(file_name, layer_name) &&
          EndsWith(file_name, PDMODEL_SUFFIX)) {
        std::string prefix = file_name.substr(
            0, file_name.length() - std::string(PDMODEL_SUFFIX).length());

        if (prefix == layer_name) {
          model_paths.emplace_back(
              std::make_pair("forward", dir_path + file_name));
        } else {
          std::string func_name = prefix.substr(layer_name.size() + 1);
          model_paths.emplace_back(
              std::make_pair(func_name, dir_path + file_name));
        }
        VLOG(3) << "func_name: " << model_paths.back().first
                << ", path:" << dir_path + file_name;
      }
    }
  }
  closedir(dir);
  return model_paths;
}

void InitKernelSignatureMap() {
  paddle::framework::InitDefaultKernelSignatureMap();
}

}  // namespace paddle::jit::utils
