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

#include "paddle/fluid/pir/serialize_deserialize/include/ir_deserialize.h"
#include "paddle/fluid/pir/serialize_deserialize/include/deserialize_utils.h"
namespace pir {
void ProgramReader::RecoverProgram(Json* program_json,
                                   pir::Program* recover_program) {
  id_value_map[0] = pir::Value();
  ReadProgram(program_json, recover_program);
  VLOG(6) << "Finish json to program.";
  return;
}
void ProgramReader::ReadProgram(Json* program_json, pir::Program* program) {
  auto top_level_op = program->module_op();
  PADDLE_ENFORCE_EQ(
      program_json->at(REGIONS).size(),
      1,
      common::errors::InvalidArgument(
          "The redions size of program module should be 1 but got %d.",
          program_json->at(REGIONS).size()));
  auto& region_json = program_json->at(REGIONS).at(0);
  auto& block_json = region_json.at(BLOCKS).at(0);
  auto& block = top_level_op.block();
  ReadBlock(&block_json, &block);

  VLOG(6) << "Finish Read program.";
  return;
}

void ProgramReader::ReadRegion(Json* region_json, pir::Region* region) {
  auto region_name = region_json->at(ID).template get<std::string>();
  for (auto& block_json : region_json->at(BLOCKS)) {
    auto& block = region->emplace_back();
    ReadBlock(&block_json, &block);
  }
  VLOG(6) << "Finish Read " << region_name;
  return;
}

void ProgramReader::ReadBlock(Json* block_json, pir::Block* block) {
  auto block_name = block_json->at(ID).template get<std::string>();

  Json& args_json = block_json->at(BLOCKARGS);
  if (!args_json.empty()) {
    for (auto& arg_json : args_json) {
      int64_t arg_id_ = arg_json.at(ID).template get<int64_t>();
      auto value = block->AddArg(ReadType(&(arg_json.at(TYPE_TYPE))));
      id_value_map[arg_id_] = value;
      VLOG(6) << "Finish Read blockargument " << arg_id_;
    }
  }

  Json& ops_json = block_json->at(BLOCKOPS);
  if (!ops_json.empty()) {
    for (auto& op_json : ops_json) {
      block->push_back(ReadOp(&op_json));
    }
  }

  VLOG(6) << "Finish Read " << block_name;
  return;
}

pir::Operation* ProgramReader::ReadOp(Json* op_json) {
  auto op_name = op_json->at(ID).template get<std::string>();
  VLOG(0) << "begin read op " << op_name;
  // deserialize opoperands (find value)
  Json& operands_json = op_json->at(OPOPERANDS);
  std::vector<pir::Value> inputs;
  for (auto& operand_json : operands_json) {
    int64_t id = operand_json.at(ID).template get<int64_t>();
    inputs.push_back(id_value_map[id]);
  }

  // deserialize opresults (find type)
  Json& opresults_json = op_json->at(OPRESULTS);
  std::vector<pir::Type> output_types;
  std::vector<int64_t> output_ids;
  for (auto& opresult_json : opresults_json) {
    int64_t value_id_ = opresult_json.at(ID).template get<int64_t>();
    output_ids.push_back(value_id_);
    output_types.push_back(ReadType(&(opresult_json.at(TYPE_TYPE))));
    VLOG(6) << "Finish Read value " << value_id_;
  }

  // serialize necessary attributes
  Json& attrs_json = op_json->at(ATTRS);

  pir::AttributeMap attributes;
  if (op_json->contains(OPRESULTS_ATTRS)) {
    Json& opresults_attrs_json = op_json->at(OPRESULTS_ATTRS);
    attributes = ReadAttributesMap(&attrs_json, &opresults_attrs_json);
  } else {
    Json empty_json = Json::array();
    attributes = ReadAttributesMap(&attrs_json, &empty_json);
  }

  pir::IrContext* ctx_ = pir::IrContext::Instance();
  // prepare opinfo
  pir::OpInfo op_info = ctx_->GetRegisteredOpInfo(op_name);
  VLOG(0) << "&&&&&&&&&&&&&&1";
  VLOG(0) << "output_type " << output_types[0];
  // deserialize op
  pir::Operation* op =
      Operation::Create(inputs, attributes, output_types, op_info);
  VLOG(0) << "&&&&&&&&&&&&&&2";
  PADDLE_ENFORCE_EQ(
      output_ids.size(),
      static_cast<size_t>(op->num_results()),
      common::errors::InvalidArgument(
          "deserialized op has %d results, but the original op has %d results.",
          op->num_results(),
          output_ids.size()));

  for (uint32_t i = 0; i < op->num_results(); i++) {
    id_value_map[output_ids[i]] = op->result(i);
  }

  VLOG(6) << "Finish Read Operation " << op->name();
  return op;
}

pir::AttributeMap ProgramReader::ReadAttributesMap(Json* attrs_json,
                                                   Json* opresult_attrs_json) {
  pir::AttributeMap attributes;
  for (auto& attr_json : *attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    VLOG(0)<<"attr_name: "<<attr_name;
    attributes.insert({attr_name, ReadAttribute(&attr_json)});
  }
  VLOG(6) << "Finish Read pir::AttributeMap ";
  for (auto& attr_json : *opresult_attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    attributes.insert({attr_name, ReadAttribute(&attr_json)});
  }
  VLOG(6) << "Finish Read Opresults_AttributeMap ";
  return attributes;
}

pir::Attribute ProgramReader::ReadAttribute(Json* attr_json) {
  VLOG(6) << "Begin Read Attribute. ";
  auto res =  pir::parseAttr(&attr_json->at(ATTR_TYPE));
  VLOG(6) << "Finish Read Attribute. ";
  return res;
}

pir::Type ProgramReader::ReadType(Json* type_json) {
  VLOG(6) << "Begin Read Type. ";
  return pir::parseType(type_json);
}

}  // namespace pir
