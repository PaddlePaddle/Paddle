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
#include <unordered_map>
#include "paddle/fluid/pir/serialize_deserialize/include/deserialize_utils.h"
namespace pir {
void ProgramReader::RecoverProgram(Json* program_json,
                                   pir::Program* recover_program,
                                   pir::PatchBuilder* builder) {
  id_value_map[0] = pir::Value();
  patch_builder = builder;
  ReadProgram(program_json, recover_program);
  VLOG(6) << "Finish json to program.";
  return;
}

pir::Type ProgramReader::RecoverType(Json* type_json) {
  return ReadType(type_json);
}

pir::AttributeMap ProgramReader::RecoverOpAttributesMap(Json* attrs_json) {
  Json empty_json = Json::array();
  std::unordered_map<std::string, Json> attr_patch;
  return ReadAttributesMap(attrs_json, &empty_json, attr_patch);
}

void ProgramReader::ReadProgram(Json* program_json, pir::Program* program) {
  auto top_level_op = program->module_op();
  PADDLE_ENFORCE_EQ(
      program_json->at(REGIONS).size(),
      1,
      common::errors::InvalidArgument(
          "The regions size of program module should be 1 but got %d.",
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
    // get value id for op_pair io patch
    VLOG(6) << "Begin to read value num ...";
    int64_t max_value_id = 0;
    for (auto& op_json : ops_json) {
      if (op_json.at(ID).template get<std::string>() == PARAMETEROP) {
        int64_t id = op_json.at(OPRESULTS).at(VALUE_ID).template get<int64_t>();
        max_value_id = std::max(max_value_id, id);
        continue;
      }
      Json& operands_json = op_json.at(OPOPERANDS);
      for (auto& operand_json : operands_json) {
        int64_t id = operand_json.at(VALUE_ID).template get<int64_t>();
        max_value_id = std::max(max_value_id, id);
      }
      Json& opresults_json = op_json.at(OPRESULTS);
      for (auto& opresult_json : opresults_json) {
        int64_t id = opresult_json.at(VALUE_ID).template get<int64_t>();
        max_value_id = std::max(max_value_id, id);
      }
    }
    max_value_id += id_value_map.size();
    VLOG(6) << "max_value_id: " << max_value_id;
    // Apply op_pair io patch
    patch_builder->ApplyOpPairPatches(&max_value_id);
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
  if (patch_builder->HasOpPatch(PARAMETEROP)) {
    VLOG(8) << PARAMETEROP << " before: " << *op_json;
    Json op_patch = patch_builder->GetJsonOpPatch(PARAMETEROP);
    VLOG(8) << " get op patch:  " << op_patch;
    patch_builder->ApplyOpPatches(PARAMETEROP, op_json, op_patch);
    VLOG(8) << PARAMETEROP << " has been patched: " << *op_json;
  }

  std::vector<pir::Value> inputs;
  Json& opresult_json = op_json->at(OPRESULTS);
  std::vector<pir::Type> output_types;

  int64_t value_id_ = opresult_json.at(VALUE_ID).template get<int64_t>();
  output_types.push_back(ReadType(&(opresult_json.at(TYPE_TYPE))));
  VLOG(6) << "Finish Read value " << value_id_ << ".";

  Json& attrs_json = op_json->at(ATTRS);
  PADDLE_ENFORCE_EQ(
      attrs_json.size(),
      4,
      common::errors::InvalidArgument(
          "builtin ParameterOp has %d 's  attributes, which should be 4",
          attrs_json.size()));
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

  if (op_json->contains(DIST_ATTRS)) {
    Json& dist_attrs_json = op_json->at(DIST_ATTRS);
    for (auto& attr_json : dist_attrs_json) {
      auto attr_name = attr_json.at(NAME).template get<std::string>();
      if (attr_json.contains(ATTR_TYPE)) {
        attributes.insert({attr_name, ReadAttribute(&attr_json)});
      } else {
        VLOG(6) << "Attribute " << attr_name << " Deleted.";
      }
    }
  }
  if (op_json->contains(QUANT_ATTRS)) {
    Json& quant_attrs_json = op_json->at(QUANT_ATTRS);
    for (auto& attr_json : quant_attrs_json) {
      auto attr_name = attr_json.at(NAME).template get<std::string>();
      if (attr_json.contains(ATTR_TYPE)) {
        attributes.insert({attr_name, ReadAttribute(&attr_json)});
      } else {
        VLOG(6) << "Attribute " << attr_name << " Deleted.";
      }
    }
  }

  if (op_json->contains(OPRESULTS_ATTRS)) {
    Json& other_attrs_json = op_json->at(OPRESULTS_ATTRS);
    PADDLE_ENFORCE_EQ(other_attrs_json.size(),
                      3,
                      common::errors::InvalidArgument(
                          "builtin ParameterOp has %d 's  opresult attributes, "
                          "which should be 3",
                          other_attrs_json.size()));
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
  pir::Operation* op = Operation::Create(
      inputs, attributes, output_types, op_info, 0, {}, false);

  id_value_map[value_id_] = op->result(0);
  VLOG(4) << "Finish Read Operation " << op->name() << ".";
  return op;
}

pir::Operation* ProgramReader::ReadOp(Json* op_json) {
  // deal with patches
  auto op_name = op_json->at(ID).template get<std::string>();
  std::unordered_map<std::string, Json> attr_patch;
  if (op_name == PARAMETEROP) {
    return ReadParameterOp(op_json);
  }
  if (patch_builder->HasOpPatch(op_name)) {
    VLOG(8) << op_name << " before: " << *op_json;
    Json op_patch = patch_builder->GetJsonOpPatch(op_name);
    VLOG(8) << " get op patch:  " << op_patch;
    attr_patch = patch_builder->GetOpAttrPatchMap(op_patch);
    VLOG(8) << " get attr_patch:  " << attr_patch;
    patch_builder->ApplyOpPatches(op_name, op_json, op_patch);
    VLOG(8) << op_name << " has been patched: " << *op_json;
    // Apply patch to op name
    // This happens when changing an op into another dialect
    if (op_patch.contains(NEW_NAME)) {
      std::string new_name = op_patch.at(NEW_NAME).template get<std::string>();
      VLOG(8) << "change op name from " << op_name << " to " << new_name;
      op_name = new_name;
      op_json->at(ID) = op_name;
      VLOG(8) << "op_json after changing name: " << *op_json;
    }
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
  int64_t value_id_ = -1;
  for (auto& opresult_json : opresults_json) {
    if (opresult_json.contains(VALUE_ID)) {
      value_id_ = opresult_json.at(VALUE_ID).template get<int64_t>();
    }
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
    attributes =
        ReadAttributesMap(&attrs_json, &opresults_attrs_json, attr_patch);
  } else {
    Json empty_json = Json::array();
    attributes = ReadAttributesMap(&attrs_json, &empty_json, attr_patch);
  }

  pir::IrContext* ctx_ = pir::IrContext::Instance();
  // prepare opinfo
  pir::OpInfo op_info = ctx_->GetRegisteredOpInfo(op_name);

  size_t num_regions = 0;
  if (op_json->contains(REGIONS)) {
    num_regions = op_json->at(REGIONS).size();
  }
  // deserialize op
  pir::Operation* op = Operation::Create(
      inputs, attributes, output_types, op_info, num_regions, {}, false);

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
    if (output_ids[i] != -1) {
      id_value_map[output_ids[i]] = op->result(i);
    }
  }

  VLOG(4) << "Finish Read Operation " << op->name() << ".";
  return op;
}

pir::AttributeMap ProgramReader::ReadAttributesMap(
    Json* attrs_json,
    Json* opresult_attrs_json,
    const std::unordered_map<std::string, Json>& attr_patch) {
  pir::AttributeMap attributes;
  // Add new attribute from patch
  if (attr_patch.count(ADD_ATTRS)) {
    for (auto& attr_json : attr_patch.at(ADD_ATTRS)) {
      attrs_json->insert(attrs_json->end(), attr_json);
    }
    VLOG(8) << "attr has been added: " << *attrs_json;
  }
  for (auto& attr_json : *attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    if (attr_patch.count(attr_name)) {
      Json patch = attr_patch.at(attr_name);
      VLOG(8) << attr_name << " has patch: " << patch;
      patch_builder->ApplyAttrPatches(attr_name, &attr_json, patch);
      VLOG(8) << attr_name << " has been patched: " << attr_json;
    }
    // Get attr_name again after patch
    attr_name = attr_json.at(NAME).template get<std::string>();
    if (attr_json.contains(ATTR_TYPE)) {
      attributes.insert({attr_name, ReadAttribute(&attr_json)});
    } else {
      VLOG(6) << "Attribute " << attr_name << " Deleted.";
    }
  }
  VLOG(6) << "Finish Read pir::AttributeMap.";
  // Add new opresult attribute from patch
  if (attr_patch.count(ADD_OPRESULTS_ATTRS)) {
    for (auto& attr_json : attr_patch.at(ADD_OPRESULTS_ATTRS)) {
      opresult_attrs_json->insert(opresult_attrs_json->end(), attr_json);
    }
    VLOG(8) << "opresult attr has been added: " << *opresult_attrs_json;
  }
  for (auto& attr_json : *opresult_attrs_json) {
    auto attr_name = attr_json.at(NAME).template get<std::string>();
    VLOG(8) << attr_name << " patch: " << attr_patch;
    if (attr_patch.count(attr_name)) {
      Json patch = attr_patch.at(attr_name);
      VLOG(8) << attr_name << " patch: " << patch;
      VLOG(8) << attr_name << " before: " << attr_json;
      patch_builder->ApplyAttrPatches(attr_name, &attr_json, patch);
      VLOG(8) << attr_name << " has been patched: " << attr_json;
    }
    // Get attr_name again after patch
    attr_name = attr_json.at(NAME).template get<std::string>();
    if (attr_json.contains(ATTR_TYPE)) {
      attributes.insert({attr_name, ReadAttribute(&attr_json)});
    } else {
      VLOG(6) << "Attribute " << attr_name << " Deleted.";
    }
  }
  VLOG(4) << "Finish Read Opresults_AttributeMap.";
  return attributes;
}

pir::Attribute ProgramReader::ReadAttribute(Json* attr_json) {
  VLOG(6) << "Begin Read Attribute. ";
  auto attr_type = attr_json->at(ATTR_TYPE).at(ID).template get<std::string>();
  if (patch_builder && patch_builder->HasAttrPatch(attr_type)) {
    VLOG(8) << attr_type << " before: " << *attr_json;
    Json attr_patch = patch_builder->GetJsonAttrPatch(attr_type);
    patch_builder->ApplyAttrTypePatches(
        attr_type, &attr_json->at(ATTR_TYPE), attr_patch);
    VLOG(8) << attr_type << " has been patched: " << *attr_json;
  }
  return pir::parseAttr(&attr_json->at(ATTR_TYPE));
}

pir::Type ProgramReader::ReadType(Json* type_json) {
  VLOG(6) << "Begin Read Type. ";
  auto type_name = type_json->at(ID).template get<std::string>();
  VLOG(8) << "Check patches for: " << type_name;
  if (patch_builder && patch_builder->HasTypePatch(type_name)) {
    VLOG(8) << type_name << " before: " << *type_json;
    Json type_patch = patch_builder->GetJsonTypePatch(type_name);
    patch_builder->ApplyTypePatches(type_name, type_json, type_patch);
    VLOG(8) << type_name << " has been patched: " << *type_json;
  }
  return pir::parseType(type_json);
}

}  // namespace pir
