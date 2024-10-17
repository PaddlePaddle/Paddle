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

#include "paddle/fluid/pir/serialize_deserialize/include/ir_serialize.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/serialize_deserialize/include/serialize_utils.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

COMMON_DECLARE_bool(save_cf_stack_op);
namespace pir {

Json ProgramWriter::GetProgramJson(const pir::Program* program) {
  program_json = WriteProgram(program);
  VLOG(6) << "Finish program to json.";
  return program_json;
}

Json ProgramWriter::GetTypeJson(const pir::Type& type) {
  auto type_json = WriteType(type);
  VLOG(6) << "Finish type to json.";
  return type_json;
}

Json ProgramWriter::GetAttributesMapJson(const AttributeMap& attr_map) {
  Json attrs_json = Json::array();
  for (auto attr : attr_map) {
    attrs_json.emplace_back(WriteAttribute(attr.first, attr.second));
  }
  return attrs_json;
}

Json ProgramWriter::WriteProgram(const pir::Program* program) {
  Json program_json;
  program_json[REGIONS] = Json::array();
  auto top_level_op = program->module_op();

  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    std::string region_name = "region_" + std::to_string(region_id_++);
    auto& region = top_level_op->region(i);
    auto region_json = WriteRegion(&region, region_name);
    program_json[REGIONS].emplace_back(region_json);
  }
  VLOG(6) << "Finish write program.";
  return program_json;
}

Json ProgramWriter::WriteRegion(const pir::Region* region,
                                const std::string& region_name) {
  Json region_json;
  region_json[ID] = region_name;
  region_json[BLOCKS] = Json::array();
  for (auto block : region->blocks()) {
    std::string block_name = "block_" + std::to_string(block_id_++);
    auto block_json = WriteBlock(block, block_name);
    region_json[BLOCKS].emplace_back(block_json);
  }
  VLOG(6) << "Finish write " << region_name;
  return region_json;
}

Json ProgramWriter::WriteBlock(pir::Block* block,
                               const std::string& block_name) {
  Json block_json;
  block_json[ID] = block_name;
  VLOG(4) << "Begin write " << block_name << ".";
  Json args_json = Json::array();
  for (auto arg : block->args()) {
    auto arg_json = WriteBlockArg(arg);
    args_json.emplace_back(arg_json);
  }
  block_json[BLOCKARGS] = args_json;
  VLOG(6) << "Finish Write blockarguments .";

  if (block->kwargs_size() > 0) {
    VLOG(6) << "Block has kwargs.";
    Json kwargs_json = Json::array();
    for (auto item : block->kwargs()) {
      auto arg_json = WriteBlockArg(item.second);
      arg_json[KEYWORDNAME] = item.first;
      kwargs_json.emplace_back(arg_json);
    }
    block_json[KEYWORDBLOCKARGS] = args_json;
    VLOG(6) << "Finish Write keyword blockarguments. ";
  }

  Json ops_json = Json::array();

  /* delete cf.stack_create / cf.tuple_push */
  if (!FLAGS_save_cf_stack_op) {
    std::vector<pir::Operation*> delete_ops;
    for (auto op : block->ops()) {
      if (op->isa<pir::StackCreateOp>()) {
        delete_ops.push_back(op);
      }
    }
    VLOG(6) << "program before delete stack op :" << *(block->parent_program());
    for (auto op : delete_ops) {
      VLOG(0) << "Delete cf.stack_create / cf.tuple_push.";
      auto stack_op = op->dyn_cast<pir::StackCreateOp>();
      if (stack_op.inlet().HasOneUse()) {
        auto tuple_push_op = stack_op.tuple_push_op();
        auto block_in = tuple_push_op->GetParent();
        block_in->erase(*tuple_push_op);
      }
      if (stack_op.outlet().HasOneUse()) {
        auto tuple_pop_op = stack_op.tuple_pop_op();
        auto block_in = tuple_pop_op->GetParent();
        block_in->erase(*tuple_pop_op);
      }
      block->erase(*op);
    }
    VLOG(6) << "program after delete stack op :" << *(block->parent_program());
  }
  for (auto op : block->ops()) {
    auto op_json = WriteOp(*op);
    ops_json.emplace_back(op_json);
  }
  block_json[BLOCKOPS] = ops_json;

  VLOG(4) << "Finish write " << block_name << ".";
  return block_json;
}

Json ProgramWriter::WriteBlockArg(const pir::Value& value) {
  Json arg_json;
  Json var = WriteType(value.type());
  value_id_map[value] = blockarg_id_;
  arg_json[ID] = blockarg_id_;
  arg_json[TYPE_TYPE] = var;

  VLOG(6) << "Finish write blockargument " << blockarg_id_ << ".";
  blockarg_id_--;

  return arg_json;
}

Json ProgramWriter::WriteValue(const pir::Value& value) {
  Json var_json;
  if (value) {
    value_id_map[value] = value_id_;
    var_json[VALUE_ID] = value_id_;
    VLOG(6) << "Finish write value " << value_id_ << ".";
    value_id_++;
  } else {
    var_json[VALUE_ID] = 0;  // NULL_TYPE
    VLOG(6) << "Finish write NULL_TYPE value.";
  }

  Json var = WriteType(value.type());
  var_json[TYPE_TYPE] = var;

  return var_json;
}
#define OPTIONAL_CHECK(array_json, attr_name, int)          \
  if (op.attributes().count(attr_name) > 0) {               \
    array_json.emplace_back(                                \
        ONE_BOOL_ARRAY_ATTRIBUTE_CAST_TEMPLATE(attr_name)); \
  } else {                                                  \
    array_json.emplace_back(int);                           \
  }

#define ONE_BOOL_ARRAY_ATTRIBUTE_CAST_TEMPLATE(attr_name)   \
  static_cast<int32_t>(op.attributes()                      \
                           .at(attr_name)                   \
                           .dyn_cast<pir::ArrayAttribute>() \
                           .at(0)                           \
                           .dyn_cast<pir::BoolAttribute>()  \
                           .data())
Json ProgramWriter::WriteParameterOP(const pir::Operation& op) {
  std::vector<std::string> AttrsNameList = {"is_distributed",
                                            "is_parameter",
                                            "need_clip",
                                            "parameter_name",
                                            "persistable",
                                            "stop_gradient",
                                            "trainable",
                                            "op_callstack" /*no need*/};
  std::vector<std::string> DistAttrsNameList = GetOpDistAttr();
  std::vector<std::string> QuantAttrsNameList = GetOpQuantAttr();
  AttrsNameList.insert(
      AttrsNameList.end(), DistAttrsNameList.begin(), DistAttrsNameList.end());
  AttrsNameList.insert(AttrsNameList.end(),
                       QuantAttrsNameList.begin(),
                       QuantAttrsNameList.end());
  for (auto attr : op.attributes()) {
    auto attr_name = attr.first;
    auto it = std::find(AttrsNameList.begin(), AttrsNameList.end(), attr_name);
    if (it == AttrsNameList.end()) {
      PADDLE_ENFORCE(
          false,
          common::errors::InvalidArgument(
              "attr name %s not supposed be serialized in WriteParameterOP, "
              "please add it in order and add deserialization code in "
              "ReadParameterOP.",
              attr_name));
    }
  }
  // attr_name ; type
  // is_distributed; array(bool)
  // is_parameter; array(bool)
  // need_clip; array(bool)
  // parameter_name; string
  // persistable; array(bool)
  // stop_gradient; array(bool)
  // trainable; array(bool)
  Json op_json = Json::object();
  op_json[ID] = PARAMETEROP;
  // serialize opoperands
  VLOG(4) << "Begin write Operation " << op.name() << ".";
  op_json[OPRESULTS] = WriteValue(op.result(0));
  Json attrs_json = Json::array();
  OPTIONAL_CHECK(attrs_json, "is_distributed", 0)
  OPTIONAL_CHECK(attrs_json, "is_parameter", 1)
  OPTIONAL_CHECK(attrs_json, "need_clip", 0)

  if (op.attributes().count("parameter_name") > 0) {
    attrs_json.emplace_back(op.attributes()
                                .at("parameter_name")
                                .dyn_cast<pir::StrAttribute>()
                                .AsString());
  } else {
    PADDLE_ENFORCE(false,
                   common::errors::InvalidArgument(
                       "parameter_name not found in ParameterOp"));
  }
  op_json[ATTRS] = attrs_json;

  Json dist_attrs_json = Json::array();
  for (auto key : GetOpDistAttr()) {
    if (op.attributes().count(key) > 0) {
      dist_attrs_json.emplace_back(
          WriteAttribute(key, op.attributes().at(key)));
    }
  }
  op_json[DIST_ATTRS] = dist_attrs_json;

  Json quant_attrs_json = Json::array();
  for (auto key : GetOpQuantAttr()) {
    if (op.attributes().count(key) > 0) {
      quant_attrs_json.emplace_back(
          WriteAttribute(key, op.attributes().at(key)));
    }
  }
  op_json[QUANT_ATTRS] = quant_attrs_json;

  Json other_attrs_json = Json::array();
  OPTIONAL_CHECK(other_attrs_json, "persistable", 1)
  OPTIONAL_CHECK(other_attrs_json, "stop_gradient", 1)
  OPTIONAL_CHECK(other_attrs_json, "trainable", 1)
  if (trainable_) {
    op_json[OPRESULTS_ATTRS] = other_attrs_json;
  }
  return op_json;
}
Json ProgramWriter::WriteOp(const pir::Operation& op) {
  if (op.isa<pir::ParameterOp>()) {
    return WriteParameterOP(op);
  }
  Json op_json = Json::object();
  auto op_name = op.name();
  GetCompressOpName(&op_name);
  op_json[ID] = op_name;
  // serialize opoperands
  VLOG(4) << "Begin write Operation " << op.name() << ".";
  Json operands_json = Json::array();
  for (auto operand : op.operands()) {
    auto operand_json = WriteOpOperand(operand);
    operands_json.emplace_back(operand_json);
  }
  op_json[OPOPERANDS] = operands_json;
  VLOG(6) << "Finish write OP's OpOperand.";
  // serialize opresults
  Json opresults_json = Json::array();
  for (auto& opresult : op.results()) {
    auto opresult_json = WriteValue(opresult);
    opresults_json.emplace_back(opresult_json);
  }
  op_json[OPRESULTS] = opresults_json;
  VLOG(6) << "Finish write OP's OpOpresult.";

  if (op.num_regions() > 0) {
    VLOG(4) << "OP has " << op.num_regions() << " regions ...";
    for (size_t i = 0; i < op.num_regions(); ++i) {
      std::string region_name = "region_" + std::to_string(region_id_++);
      auto& region = op.region(i);
      auto region_json = WriteRegion(&region, region_name);
      op_json[REGIONS].emplace_back(region_json);
    }
    VLOG(4) << "Finish write OP's regions.";
  }
  // serialize attributes
  op_json[ATTRS] = WriteAttributesMapOpinfo(const_cast<pir::Operation*>(&op),
                                            op.attributes());

  if (trainable_) {
    op_json[OPRESULTS_ATTRS] = WriteAttributesMapOther(op.attributes());
  }

  VLOG(4) << "Finish write Operation " << op.name() << ".";
  return op_json;
}

Json ProgramWriter::WriteOpOperand(const pir::OpOperand& op_operand) {
  Json operand_json = Json::object();
  if (op_operand.source()) {
    int64_t id = value_id_map[op_operand.source()];
    operand_json[VALUE_ID] = id;
    VLOG(6) << "Finish write OpOperand " << id << ".";
  } else {
    operand_json[VALUE_ID] = 0;  // NULL_VALUE
    VLOG(6) << "Finish write NULL_VALUE OpOperand.";
  }

  return operand_json;
}

Json ProgramWriter::WriteAttributesMapOpinfo(pir::Operation* op,
                                             const AttributeMap& attr_map) {
  Json attrs_json = Json::array();
  VLOG(6) << "Start write Opinfo AttributeMap ...";
  if (op->dialect()->name() == "pd_op" &&
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>()) {
    auto [_1, attr_info, _3, _4, _5] =
        op->dyn_cast<paddle::dialect::OpYamlInfoInterface>().GetOpInfo();
    if (attr_info.size() != 0) {
      for (const auto& val : attr_info) {
        if (attr_map.find(val.name) != attr_map.end()) {
          attrs_json.emplace_back(
              WriteAttribute(val.name, attr_map.at(val.name)));
        }
      }
    }
    for (auto key : GetOpDistAttr()) {
      if (attr_map.count(key) > 0) {
        attrs_json.emplace_back(WriteAttribute(key, attr_map.at(key)));
      }
    }
    for (auto key : GetOpQuantAttr()) {
      if (attr_map.count(key) > 0) {
        attrs_json.emplace_back(WriteAttribute(key, attr_map.at(key)));
      }
    }
  } else {
    for (auto& attr : attr_map) {
      if (attr.first != "stop_gradient" && attr.first != "persistable" &&
          attr.first != "op_callstack") {
        attrs_json.emplace_back(WriteAttribute(attr.first, attr.second));
      }
    }
  }

  VLOG(6) << "Finish write Opinfo AttributeMap. ";
  return attrs_json;
}

Json ProgramWriter::WriteAttributesMapOther(const AttributeMap& attr_map) {
  Json operesult_attrs_json = Json::array();
  for (auto& attr : attr_map) {
    if (attr.first == "stop_gradient" || attr.first == "persistable") {
      operesult_attrs_json.emplace_back(
          WriteAttribute(attr.first, attr.second));
    }
  }

  VLOG(6) << "Finish write Other AttributeMap. ";
  return operesult_attrs_json;
}

Json ProgramWriter::WriteAttribute(const std::string& op_attr_name,
                                   const pir::Attribute& attr) {
  Json attr_json;
  attr_json[NAME] = op_attr_name;
  attr_json[ATTR_TYPE] = pir::writeAttr(attr);

  VLOG(6) << "Finish write Attribute. ";
  return attr_json;
}

Json ProgramWriter::WriteType(const pir::Type& type) {
  VLOG(6) << "Finish write Type. ";
  return pir::writeType(type);
}
}  // namespace pir
