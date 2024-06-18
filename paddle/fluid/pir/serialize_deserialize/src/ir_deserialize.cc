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
    region->emplace_back();
    ReadBlock(&block_json, &(region->back()));
  }
  VLOG(6) << "Finish Read " << region_name << ".";
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
      VLOG(6) << "Finish Read blockargument " << arg_id_ << ".";
    }
    VLOG(6) << "Finish Read blockarguments.";
  }

  if (block_json->contains(KEYWORDBLOCKARGS)) {
    Json& kwargs_json = block_json->at(KEYWORDBLOCKARGS);
    if (!kwargs_json.empty()) {
      for (auto& kwarg_json : kwargs_json) {
        int64_t arg_id_ = kwarg_json.at(ID).template get<int64_t>();
        auto value = block->AddKwarg(kwarg_json.at(KEYWORDNAME),
                                     ReadType(&(kwarg_json.at(TYPE_TYPE))));
        id_value_map[arg_id_] = value;
        VLOG(6) << "Finish Read keyword blockargument " << arg_id_ << ".";
      }
      VLOG(6) << "Finish Read keyword blockarguments. ";
    }
  }

  Json& ops_json = block_json->at(BLOCKOPS);
  if (!ops_json.empty()) {
    for (auto& op_json : ops_json) {
      block->push_back(ReadOp(&op_json));
    }
    VLOG(6) << "read block size" << block->size() << ".";
  }

  VLOG(4) << "Finish Read " << block_name << ".";
  return;
}
pir::ArrayAttribute GetOneBoolArrayAttribute(pir::IrContext* ctx,
                                             Json* attr_json) {
  std::vector<pir::Attribute> val;
  bool bool_value = attr_json->template get<int32_t>() != 0;
  val.push_back(pir::BoolAttribute::get(ctx, bool_value));
  return pir::ArrayAttribute::get(ctx, val);
}

pir::Operation* ProgramReader::ReadParameterOp(Json* op_json) {
  // attr is_distributed; is_parameter; need_clip; parameter_name; persistable;
  // stop_gradient; trainable;
  std::vector<pir::Value> inputs;
  Json& opresult_json = op_json->at(OPRESULTS);
  std::vector<pir::Type> output_types;

  int64_t value_id_ = opresult_json.at(VALUE_ID).template get<int64_t>();
  output_types.push_back(ReadType(&(opresult_json.at(TYPE_TYPE))));
  VLOG(6) << "Finish Read value " << value_id_ << ".";

  Json& attrs_json = op_json->at(ATTRS);
  pir::AttributeMap attributes;
  pir::IrContext* ctx = pir::IrContext::Instance();
  attributes.insert(
      {"is_distributed", GetOneBoolArrayAttribute(ctx, &attrs_json.at(0))});
  attributes.insert(
      {"is_parameter", GetOneBoolArrayAttribute(ctx, &attrs_json.at(1))});
  attributes.insert(
      {"need_clip", GetOneBoolArrayAttribute(ctx, &attrs_json.at(2))});
  attributes.insert({"parameter_name",
                     pir::StrAttribute::get(
                         ctx, attrs_json.at(3).template get<std::string>())});

  if (op_json->contains(OPRESULTS_ATTRS)) {
    Json& other_attrs_json = op_json->at(OPRESULTS_ATTRS);
    attributes.insert({"persistable",
                       GetOneBoolArrayAttribute(ctx, &other_attrs_json.at(0))});
    attributes.insert({"stop_gradient",
                       GetOneBoolArrayAttribute(ctx, &other_attrs_json.at(1))});
    attributes.insert(
        {"trainable", GetOneBoolArrayAttribute(ctx, &other_attrs_json.at(2))});
  }

  pir::IrContext* ctx_ = pir::IrContext::Instance();
  // prepare opinfo
  pir::OpInfo op_info = ctx_->GetRegisteredOpInfo(pir::ParameterOp::name());
  // deserialize op
  pir::Operation* op =
      Operation::Create(inputs, attributes, output_types, op_info);

  id_value_map[value_id_] = op->result(0);
  VLOG(4) << "Finish Read Operation " << op->name() << ".";
  return op;
}

pir::Operation* ProgramReader::ReadOp(Json* op_json) {
  auto op_name = op_json->at(ID).template get<std::string>();
  if (op_name == PARAMETEROP) {
    return ReadParameterOp(op_json);
  }
  GetDecompressOpName(&op_name);
  VLOG(4) << "Read op_name = " << op_name << ".";
  // deserialize opoperands (find value)
  Json& operands_json = op_json->at(OPOPERANDS);
  std::vector<pir::Value> inputs;
  for (auto& operand_json : operands_json) {
    int64_t id = operand_json.at(VALUE_ID).template get<int64_t>();
    inputs.push_back(id_value_map[id]);
  }
  VLOG(6) << "Finish Read OP's OpOperand.";
  // deserialize opresults (find type)
  Json& opresults_json = op_json->at(OPRESULTS);
  std::vector<pir::Type> output_types;
  std::vector<int64_t> output_ids;
  for (auto& opresult_json : opresults_json) {
    int64_t value_id_ = opresult_json.at(VALUE_ID).template get<int64_t>();
    output_ids.push_back(value_id_);
    output_types.push_back(ReadType(&(opresult_json.at(TYPE_TYPE))));
    VLOG(6) << "Finish Read value " << value_id_ << ".";
  }
  VLOG(6) << "Finish Read OP's OpResult.";

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

  size_t num_regions = 0;
  if (op_json->contains(REGIONS)) {
    num_regions = op_json->at(REGIONS).size();
  }
  // deserialize op
  pir::Operation* op =
      Operation::Create(inputs, attributes, output_types, op_info, num_regions);

  // deserialize op's regions
  if (op_json->contains(REGIONS)) {
    Json& regions_json = op_json->at(REGIONS);
    VLOG(6) << op->name() << " has " << num_regions << " regions.";
    for (uint64_t i = 0; i < regions_json.size(); i++) {
      auto region_json = regions_json.at(i);
      ReadRegion(&region_json, &(op->region(i)));
    }
    VLOG(6) << "Finish Read OP's regions.";
  }

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

  VLOG(4) << "Finish Read Operation " << op->name() << ".";
  return op;
}

pir::AttributeMap ProgramReader::ReadAttributesMap(Json* attrs_json,
                                                   Json* opresult_attrs_json) {
  pir::AttributeMap attributes;
  for (auto& attr_json : *attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    attributes.insert({attr_name, ReadAttribute(&attr_json)});
  }
  VLOG(6) << "Finish Read pir::AttributeMap.";
  for (auto& attr_json : *opresult_attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    attributes.insert({attr_name, ReadAttribute(&attr_json)});
  }
  VLOG(4) << "Finish Read Opresults_AttributeMap.";
  return attributes;
}

pir::Attribute ProgramReader::ReadAttribute(Json* attr_json) {
  VLOG(6) << "Begin Read Attribute. ";
  return pir::parseAttr(&attr_json->at(ATTR_TYPE));
}

pir::Type ProgramReader::ReadType(Json* type_json) {
  VLOG(6) << "Begin Read Type. ";
  return pir::parseType(type_json);
}

}  // namespace pir
