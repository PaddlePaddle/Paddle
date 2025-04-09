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

#include "paddle/fluid/framework/program_converter.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_proto.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/version.h"

namespace paddle::framework {

using paddle::experimental::ExtractPlainVector;
using paddle::experimental::WrapAsScalars;

static std::unordered_set<std::string> needConvertedOperators = {
    "assign_value", "set_value", "set_value_grad"};

std::pair<bool, std::unordered_multimap<std::string, OpDesc*>> DetectLegacyOps(
    ProgramDesc* program) {
  bool is_legacy_program = false;
  std::unordered_multimap<std::string, OpDesc*> legacy_op_map;
  std::unordered_map<std::string, uint32_t> program_op_versions;
  std::unordered_map<std::string, uint32_t> legacy_op_versions;

  // get *all kinds* of formats of op versions and op version map to a unified
  // representation before comparison can be done in a neat way
  legacy_op_versions = paddle::framework::compatible::pb::GetLegacyOpVersions();

  const auto* _op_version_map = program->OpVersionMap();
  for (int i = 0; i < _op_version_map->pair_size(); ++i) {
    auto pair = std::make_pair(
        _op_version_map->pair(i).op_name(),
        static_cast<uint32_t>(_op_version_map->pair(i).op_version().version()));
    program_op_versions.insert(pair);
  }

  const size_t num_blocks = program->Size();
  for (size_t i = 0; i < num_blocks; i++) {
    BlockDesc* block = program->MutableBlock(i);
    const size_t num_ops = block->OpSize();
    for (size_t j = 0; j < num_ops; j++) {
      OpDesc* op = block->Op(static_cast<int>(j));
      const std::string& op_type = op->Type();
      if (needConvertedOperators.find(op_type) !=
          needConvertedOperators.end()) {
        // If an operator (program_op) is in the needConvertedOperators set,
        // it indicates that the operator may need to be converted.
        // Further judgement: if the operator does not exist in the
        // program_op_version_map, the operator needs to be converted.
        // Moreover, if the operator does exist and its program_op_version_
        // is less than or equal legacy_op_version, the operator also needs to
        // be converted.
        if (!program_op_versions.count(op_type) ||
            program_op_versions[op_type] <= legacy_op_versions[op_type]) {
          is_legacy_program = true;
          legacy_op_map.insert(std::make_pair(op_type, op));
        }
      }
    }
  }
  return std::make_pair(is_legacy_program, legacy_op_map);
}

}  // namespace paddle::framework
namespace paddle::framework::no_scalar {
void ConvertSetValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values = PADDLE_GET_CONST(
      std::vector<paddle::experimental::Scalar>, op->GetAttr("values", false));
  op->RemoveAttr("values");
  op->SetAttr("bool_values", std::vector<int>());
  op->SetAttr("fp32_values", std::vector<float>());
  op->SetAttr("int32_values", std::vector<int>());
  op->SetAttr("int64_values", std::vector<int64_t>());
  op->SetAttr("fp64_values", std::vector<double>());
  op->SetAttr("fp16_values", std::vector<float>());

  phi::DataType dtype = phi::DataType::FLOAT32;
  if (!values.empty()) {
    dtype = values.at(0).dtype();
  }

  switch (dtype) {
    case phi::DataType::BOOL:
      op->SetAttr("bool_values", ExtractPlainVector<int>(values));
      break;
    case phi::DataType::FLOAT32:
      op->SetAttr("fp32_values", ExtractPlainVector<float>(values));
      break;
    case phi::DataType::INT32:
      op->SetAttr("int32_values", ExtractPlainVector<int>(values));
      break;
    case phi::DataType::INT64:
      op->SetAttr("int64_values", ExtractPlainVector<int64_t>(values));
      break;
    case phi::DataType::FLOAT64:
      op->SetAttr("fp64_values", ExtractPlainVector<double>(values));
      break;
    case phi::DataType::FLOAT16:
      op->SetAttr("fp16_values", ExtractPlainVector<float>(values));
      break;
    default:
      PD_THROW("Invalid data type `", dtype, "`.");
  }
}

void ConvertAssignValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values = PADDLE_GET_CONST(
      std::vector<paddle::experimental::Scalar>, op->GetAttr("values", false));
  op->RemoveAttr("values");
  op->SetAttr("bool_values", std::vector<int>());
  op->SetAttr("fp32_values", std::vector<float>());
  op->SetAttr("int32_values", std::vector<int>());
  op->SetAttr("int64_values", std::vector<int64_t>());

  phi::DataType dtype = phi::DataType::FLOAT32;
  if (!values.empty()) {
    dtype = values.at(0).dtype();
  }

  switch (dtype) {
    case phi::DataType::BOOL:
      op->SetAttr("bool_values", ExtractPlainVector<int>(values));
      break;
    case phi::DataType::FLOAT32:
      op->SetAttr("fp32_values", ExtractPlainVector<float>(values));
      break;
    case phi::DataType::FLOAT64:
      op->SetAttr("fp32_values", ExtractPlainVector<float>(values));
      break;
    case phi::DataType::INT32:
      op->SetAttr("int32_values", ExtractPlainVector<int>(values));
      break;
    case phi::DataType::INT64:
      op->SetAttr("int64_values", ExtractPlainVector<int64_t>(values));
      break;
    default:
      PD_THROW("Invalid data type `", dtype, "`.");
  }
}

void ConvertProgram(ProgramDesc* program) {
  PADDLE_ENFORCE_NOT_NULL(
      program, common::errors::InvalidArgument("program should not be null"));

  VLOG(3) << "Setting Program Version and OpVersionMap to Legacy "
             "settings(a.k.a 2.4.2)";
  framework::compatible::pb::OpVersionMap op_version_map(
      program->OpVersionMap());
  program->SetVersion(paddle::framework::kLegacyProgramVersion);
  paddle::framework::compatible::SaveOpVersions(
      paddle::framework::compatible::pb::GetLegacyOpVersions(),
      &op_version_map);

  VLOG(3) << "Converting program from new(with scalar attributes) to old(no "
             "scalar attributes)";

  const size_t num_blocks = program->Size();
  for (size_t i = 0; i < num_blocks; i++) {
    BlockDesc* block = program->MutableBlock(i);
    const size_t num_ops = block->OpSize();
    for (size_t j = 0; j < num_ops; j++) {
      OpDesc* op = block->Op(static_cast<int>(j));
      const std::string op_type = op->Type();
      if (op_type == "set_value" || op_type == "set_value_grad") {
        ConvertSetValueOp(op);
      }
      if (op_type == "assign_value") {
        ConvertAssignValueOp(op);
      }
    }
  }
}
}  // namespace paddle::framework::no_scalar

namespace paddle::framework::scalar {
void ConvertSetValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values;

  if (op->HasAttr("bool_values")) {
    std::vector<int> bool_values =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values", false));
    if (!bool_values.empty()) {
      values = WrapAsScalars(bool_values);
    }
    op->RemoveAttr("bool_values");
  }
  if (op->HasAttr("fp32_values")) {
    std::vector<float> fp32_values =
        PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values", false));
    if (!fp32_values.empty()) {
      values = WrapAsScalars(fp32_values);
    }
    op->RemoveAttr("fp32_values");
  }
  if (op->HasAttr("int32_values")) {
    std::vector<int> int32_values =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values", false));
    if (!int32_values.empty()) {
      values = WrapAsScalars(int32_values);
    }
    op->RemoveAttr("int32_values");
  }
  if (op->HasAttr("int64_values")) {
    std::vector<int64_t> int64_values = PADDLE_GET_CONST(
        std::vector<int64_t>, op->GetAttr("int64_values", false));
    if (!int64_values.empty()) {
      values = WrapAsScalars(int64_values);
    }
    op->RemoveAttr("int64_values");
  }
  if (op->HasAttr("fp64_values")) {
    std::vector<double> fp64_values = PADDLE_GET_CONST(
        std::vector<double>, op->GetAttr("fp64_values", false));
    if (!fp64_values.empty()) {
      values = WrapAsScalars(fp64_values);
    }
    op->RemoveAttr("fp64_values");
  }
  if (op->HasAttr("fp16_values")) {
    std::vector<float> fp16_values =
        PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp16_values", false));
    if (!fp16_values.empty()) {
      values = WrapAsScalars(fp16_values);
    }
    op->RemoveAttr("fp16_values");
  }
  op->SetAttr("values", values);
}

void ConvertAssignValueOp(OpDesc* op) {
  VLOG(3) << "convert old assign value op to new";
  std::vector<paddle::experimental::Scalar> values;

  if (op->HasAttr("bool_values")) {
    std::vector<int> bool_values =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values", false));
    if (!bool_values.empty()) {
      values = WrapAsScalars(bool_values);
    }
    op->RemoveAttr("bool_values");
  }
  if (op->HasAttr("fp32_values")) {
    std::vector<float> fp32_values =
        PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values", false));
    if (!fp32_values.empty()) {
      values = WrapAsScalars(fp32_values);
    }
    op->RemoveAttr("fp32_values");
  }
  if (op->HasAttr("int32_values")) {
    std::vector<int> int32_values =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values", false));
    if (!int32_values.empty()) {
      values = WrapAsScalars(int32_values);
    }
    op->RemoveAttr("int32_values");
  }
  if (op->HasAttr("int64_values")) {
    std::vector<int64_t> int64_values = PADDLE_GET_CONST(
        std::vector<int64_t>, op->GetAttr("int64_values", false));
    if (!int64_values.empty()) {
      values = WrapAsScalars(int64_values);
    }
    op->RemoveAttr("int64_values");
  }
  if (!values.empty()) op->SetAttr("values", values);
}

void ConvertProgram(ProgramDesc* program) {
  PADDLE_ENFORCE_NOT_NULL(
      program, common::errors::InvalidArgument("program should not be null"));

  auto legacy_op_results = DetectLegacyOps(program);
  bool is_legacy_program = legacy_op_results.first;
  const std::unordered_multimap<std::string, OpDesc*>& legacy_ops =
      legacy_op_results.second;

  VLOG(3) << "is_legacy_program : " << is_legacy_program;
  if (!is_legacy_program) return;

  VLOG(3) << "Updating Program Version and OpVersionMap";
  program->SetVersion(paddle::framework::kCurProgramVersion);
  framework::compatible::pb::OpVersionMap op_version_map(
      program->OpVersionMap());
  paddle::framework::compatible::SaveOpVersions(
      framework::compatible::get_op_version_map(), &op_version_map);

  VLOG(3) << "Converting program from old(no scalar attributes) to new(with "
             "scalar attributes)";

  for (const auto& pair : legacy_ops) {
    const std::string op_type = pair.first;
    VLOG(3) << "Converting program from old to new, op_type=" << op_type;
    if (op_type == "set_value" || op_type == "set_value_grad") {
      ConvertSetValueOp(pair.second);
    }
    if (op_type == "assign_value") {
      ConvertAssignValueOp(pair.second);
    }
  }
}
}  // namespace paddle::framework::scalar
