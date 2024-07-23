// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/pir/include/core/dll_decl.h"

namespace pir {
class PatchBuilder {
 public:
  PatchBuilder() {}
  explicit PatchBuilder(const uint64_t pir_version)
      : pir_version_(pir_version) {}
  ~PatchBuilder() = default;
  PatchBuilder(PatchBuilder&&) = delete;
  PatchBuilder(const PatchBuilder& PatchBuilder) = delete;
  PatchBuilder& operator=(const PatchBuilder&) = delete;
  PatchBuilder& operator=(PatchBuilder&&);

  void IR_API BuildPatch(const std::string& path);
  bool HasOpPatch(const std::string& name) const {
    return op_patches_.count(name) != 0;
  }
  bool HasTypePatch(const std::string& name) const {
    return type_patches_.count(name) != 0;
  }
  bool HasAttrPatch(const std::string& name) const {
    return attr_patches_.count(name) != 0;
  }
  Json GetJsonOpPatch(const std::string& name) { return op_patches_[name]; }
  Json GetJsonTypePatch(const std::string& name) { return type_patches_[name]; }
  Json GetJsonAttrPatch(const std::string& name) { return attr_patches_[name]; }
  std::unordered_map<std::string, Json> GetOpAttrPatchMap(const Json op_patch);
  void ApplyOpPatches(const std::string& op_name, Json* json, Json patch);
  void ApplyTypePatches(const std::string& type_name, Json* json, Json patch);
  void ApplyAttrPatches(const std::string& attr_name, Json* json, Json patch);
  void ApplyAttrTypePatches(const std::string& attr_name,
                            Json* json,
                            Json patch);

 private:
  uint64_t file_version_;
  uint64_t pir_version_;
  std::unordered_map<std::string, Json> op_patches_;
  std::unordered_map<std::string, Json> type_patches_;
  std::unordered_map<std::string, Json> attr_patches_;
  Json patch_json;
};

}  // namespace pir
