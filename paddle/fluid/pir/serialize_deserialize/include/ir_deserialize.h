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

#include <fstream>
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/fluid/pir/serialize_deserialize/include/version_compat.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"

namespace pir {

class ProgramReader {
 public:
  explicit ProgramReader(const uint64_t version) : current_version(version) {}

  ProgramReader(ProgramReader&&) = delete;
  ProgramReader(const ProgramReader& ProgramReader) = delete;
  ProgramReader& operator=(const ProgramReader&) = delete;
  ProgramReader& operator=(ProgramReader&&);

  // static void staticInit()

  void IR_API RecoverProgram(Json* program_json,
                             pir::Program* recover_program,
                             pir::PatchBuilder* builder);
  pir::Type RecoverType(Json* type_json);
  pir::AttributeMap RecoverOpAttributesMap(Json* attrs_json);
  ~ProgramReader() = default;

 private:
  uint64_t current_version;
  std::map<int64_t, pir::Value> id_value_map;
  pir::PatchBuilder* patch_builder = nullptr;

  void ReadProgram(Json* program_json, pir::Program* program);
  void ReadRegion(Json* region_json, pir::Region* region);
  void ReadBlock(Json* block_json, pir::Block* block);
  pir::Operation* ReadOp(Json* op_json);
  pir::AttributeMap ReadAttributesMap(
      Json* attrs_json,
      Json* operesult_attrs_json,
      const std::unordered_map<std::string, Json>& attr_patch);
  pir::Attribute ReadAttribute(Json* attr_json);
  pir::Type ReadType(Json* type_json);

  pir::Operation* ReadParameterOp(Json* op_json);
};

}  // namespace pir
