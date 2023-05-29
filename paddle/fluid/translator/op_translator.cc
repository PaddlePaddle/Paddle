// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/translator/op_translator.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/translator/attribute_translator.h"
#include "paddle/fluid/translator/op_compat_info.h"
#include "paddle/fluid/translator/program_translator.h"
#include "paddle/fluid/translator/type_translator.h"
#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/operation.h"
#include "paddle/ir/value.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace translator {

namespace {

using ResultIdx = size_t;
using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using OpOutputTypeList = std::vector<ir::Type>;
using OpOutputMapping = std::unordered_map<std::string, ResultIdx>;

static const char kTargetDialectPrefix[] = "pd.";

static const std::unordered_set<std::string> special_inplace_ops = {
    "batch_norm",
};

inline bool IsInplace(const OpDesc& op_desc) {
  bool inplace = false;
  if (special_inplace_ops.count(op_desc.Type())) {
    return inplace;
  }
  auto input_names = op_desc.InputArgumentNames();
  auto output_names = op_desc.OutputArgumentNames();

  std::vector<std::string> name_intersection;
  std::set_intersection(input_names.begin(),
                        input_names.end(),
                        output_names.begin(),
                        output_names.end(),
                        std::back_inserter(name_intersection));

  if (name_intersection.size() > 0) {
    std::string redundant_variables = std::accumulate(
        std::next(name_intersection.begin()),
        name_intersection.end(),
        name_intersection[0],
        [](std::string a, std::string b) { return a + "," + b; });
    VLOG(4) << "Following variables occur both in inputs and outputs: "
            << redundant_variables;
    return true;
  }

  return inplace;
}

inline std::string OpNamecompatibleMapping(std::string op_name) {
  auto& op_normalizer = OpNameNormalizer::instance();
  return op_normalizer[op_name];
}

inline ir::OpInfo LoopkUpOpInfo(ir::IrContext* ctx, const OpDesc& op_desc) {
  std::string target_op_name =
      kTargetDialectPrefix + OpNamecompatibleMapping(op_desc.Type());
  if (IsInplace(op_desc)) {
    target_op_name += "_";
  }
  VLOG(6) << "[op name normalizing: " << op_desc.Type() << " to "
          << target_op_name;
  auto op_info = ctx->GetRegisteredOpInfo(target_op_name);
  if (!op_info) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Op %d should have corresponding OpInfo %d",
        op_desc.Type(),
        target_op_name));
  }

  return op_info;
}

inline ir::Operation* InsertSliceOperationForTarget(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    ir::Program* program,
    const VariableDefiningInfo& defining_info,
    const std::string& arg_name) {
  std::string slice_op_name(ir::SliceOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  std::unordered_map<std::string, ir::Attribute> op_attribute_map = {
      {"index", ir::Int32_tAttribute::get(ctx, defining_info.idx_in_vector)},
  };
  ir::VectorType src_vec_type =
      defining_info.value.type().dyn_cast<ir::VectorType>();
  ir::Operation* operation =
      ir::Operation::create({defining_info.value},
                            {src_vec_type[defining_info.idx_in_vector]},
                            op_attribute_map,
                            op_info);
  program->InsertOp(operation);
  ir::OpResult target_op_result = operation->GetResultByIndex(0);
  (*param_map)[arg_name] = VariableDefiningInfo(target_op_result);
  return operation;
}

inline ir::Operation* InsertCombineOperationForTarget(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    ir::Program* program,
    const std::vector<std::string>& args) {
  std::string combine_op_name(ir::CombineOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(combine_op_name);

  std::vector<ir::OpResult> src_values;
  std::vector<ir::Type> types_in_vec;
  for (const auto& arg_name : args) {
    auto defining_info = param_map->at(arg_name);
    src_values.push_back(defining_info.value);
    types_in_vec.push_back(defining_info.value.type());
  }
  ir::Type target_vec_type = ir::VectorType::get(ctx, types_in_vec);
  ir::Operation* operation =
      ir::Operation::create(src_values, {target_vec_type}, {}, op_info);
  program->InsertOp(operation);
  return operation;
}

inline std::vector<ir::OpResult> GenerateOperationInput(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    ir::Program* program,
    const OpDesc& op_desc) {
  std::vector<ir::OpResult> op_inputs;

  // scan all inputs to see if any of them is generated as a vector<Tensor>
  // so need an additional `SliceOp` to take it out.
  for (const auto& n : op_desc.Inputs()) {
    auto& name = n.first;
    auto& args = n.second;

    for (const auto& arg_name : args) {
      PADDLE_ENFORCE_NE(
          param_map->count(arg_name),
          0,
          platform::errors::PreconditionNotMet(
              "arg %s.%s as input should be exists before prasing %d",
              name,
              arg_name,
              op_desc.Type()));
      auto defining_info = (*param_map)[arg_name];
      if (defining_info.generated_by_vector) {
        InsertSliceOperationForTarget(
            ctx, param_map, program, defining_info, arg_name);
      }
    }
  }

  for (const auto& n : op_desc.Inputs()) {
    auto& name = n.first;
    VLOG(10) << "[input retriving]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;

    // if src type is Tensor or a Vector<Tensor> with size <= 1
    if (args.size() <= 1) {
      for (const auto& arg_name : args) {
        auto defining_info = (*param_map)[arg_name];
        op_inputs.push_back(defining_info.value);
      }

      // if src type is Vector<Tesnor> , need an additional `CombineOp` to
      // assemble them.
    } else {
      auto* combine_op =
          InsertCombineOperationForTarget(ctx, param_map, program, args);
      op_inputs.push_back(combine_op->GetResultByIndex(0));
    }
  }
  return op_inputs;
}

inline std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
    ir::IrContext* ctx, const OpDesc& op_desc) {
  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types = {};

  auto& type_translator = TypeTranslator::instance();

  const BlockDesc* block = op_desc.Block();
  for (const auto& n : op_desc.Outputs()) {
    auto& name = n.first;
    VLOG(10) << "[output translating]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;

    size_t cur_output_idx = op_output_types.size();

    // if src type is Tensor or a Vector<Tensor> with size <= 1
    if (args.size() <= 1) {
      for (const auto& arg_name : args) {
        VarDesc* var = block->FindVarRecursive(arg_name);
        VLOG(10) << "[output translating]"
                 << "[" << op_desc.Type() << "]" << name << " " << arg_name
                 << " " << var->GetType();

        ir::Type translated_var_type =
            type_translator[var->GetType()](ctx, *var);

        arg_to_idx[arg_name] = cur_output_idx;
        op_output_types.push_back(translated_var_type);
      }

      // if src type is Vector<Tesnor>
    } else {
      std::vector<ir::Type> types;
      for (const auto& arg_name : args) {
        VarDesc* var = block->FindVarRecursive(arg_name);
        VLOG(10) << "[output translating]"
                 << "[" << op_desc.Type() << "]" << name << " " << arg_name
                 << " " << var->GetType();
        ir::Type translated_var_type =
            type_translator[var->GetType()](ctx, *var);
        types.push_back(translated_var_type);
        arg_to_idx[arg_name] = cur_output_idx;
      }
      ir::Type vec_type = ir::VectorType::get(ctx, types);
      op_output_types.push_back(vec_type);
    }
  }
  return {op_output_types, arg_to_idx};
}

inline ir::AttributeMap TranslateOpAttribute(const OpDesc& op_desc) {
  auto& attribute_translator = AttributeTranslator::instance();
  ir::AttributeMap attribute_map = {};
  for (auto attr_in_op_desc : op_desc.GetAttrMap()) {
    const auto& attr_name = attr_in_op_desc.first;
    const auto& attr_value = attr_in_op_desc.second;
    VLOG(0) << "attribute in " << op_desc.Type() << " name: " << attr_name
            << " " << attr_value.index();
    ir::Attribute new_attr = attribute_translator[attr_value];
    attribute_map[attr_name] = new_attr;
    if (!new_attr) {
      VLOG(0) << "empty attribute in " << op_desc.Type()
              << " name: " << attr_name;
    } else {
      VLOG(10) << "new attribute in " << op_desc.Type()
               << " name: " << attr_name << " " << new_attr.storage();
    }
  }

  for (auto attr_in_op_desc : op_desc.GetRuntimeAttrMap()) {
    const auto& attr_name = attr_in_op_desc.first;
    const auto& attr_value = attr_in_op_desc.second;
    ir::Attribute new_attr = attribute_translator[attr_value];
    attribute_map[attr_name] = new_attr;
    if (!new_attr) {
      VLOG(0) << "empty runtime attribute in " << op_desc.Type()
              << " name: " << attr_name;
    } else {
      VLOG(10) << "new runtime attribute in " << op_desc.Type()
               << " name: " << attr_name << " " << new_attr.storage();
    }
  }

  return std::move(attribute_map);
}

inline void RecordOpResultMapping(TranslationContext* param_map,
                                  const OpDesc& op_desc,
                                  ir::Operation* operation,
                                  const OpOutputMapping& arg_to_idx) {
  for (const auto& n : op_desc.Outputs()) {
    auto& name = n.first;
    VLOG(10) << "[output recording]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;
    size_t idx_in_vector = 0;
    for (const auto& arg_name : args) {
      auto idx = arg_to_idx.at(arg_name);
      VLOG(10) << "[output recording]"
               << "[" << op_desc.Type() << "]" << arg_name << " " << idx;

      ir::OpResult value = operation->GetResultByIndex(idx);
      bool generated_by_vector = value.type().isa<ir::VectorType>();
      (*param_map)[arg_name] = VariableDefiningInfo(
          value, generated_by_vector, generated_by_vector ? idx_in_vector : -1);
      idx_in_vector++;
    }
  }
}

ir::Operation* GeneralOpHandler(ir::IrContext* ctx,
                                TranslationContext* param_map,
                                ir::Program* program,
                                const OpDesc& op_desc) {
  auto op_inputs = GenerateOperationInput(ctx, param_map, program, op_desc);

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types;
  std::tie(op_output_types, arg_to_idx) = GenerateOperationOutput(ctx, op_desc);
  auto op_info = LoopkUpOpInfo(ctx, op_desc);
  auto attribute_map = TranslateOpAttribute(op_desc);
  VLOG(4) << "[general op][" << op_desc.Type() << "] preparation end.";

  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, attribute_map, op_info);
  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation creation end.";
  program->InsertOp(operation);

  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation insertion end.";
  RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

  return operation;
}

ir::Operation* FeedOpHandler(ir::IrContext* ctx,
                             TranslationContext* param_map,
                             ir::Program* program,
                             const OpDesc& op_desc) {
  std::vector<ir::OpResult> op_inputs;

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types;
  std::tie(op_output_types, arg_to_idx) = GenerateOperationOutput(ctx, op_desc);
  auto op_info = LoopkUpOpInfo(ctx, op_desc);
  ir::AttributeMap attribute_map = {
      {"name", ir::StrAttribute::get(ctx, op_desc.OutputArgumentNames()[0])},
  };

  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, attribute_map, op_info);
  program->InsertOp(operation);
  RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

  return operation;
}

ir::Operation* FetchOpHandler(ir::IrContext* ctx,
                              TranslationContext* param_map,
                              ir::Program* program,
                              const OpDesc& op_desc) {
  auto op_inputs = GenerateOperationInput(ctx, param_map, program, op_desc);

  OpOutputTypeList op_output_types;
  auto op_info = LoopkUpOpInfo(ctx, op_desc);
  ir::AttributeMap attribute_map = {
      {"name", ir::StrAttribute::get(ctx, op_desc.InputArgumentNames()[0])},
  };

  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, attribute_map, op_info);
  program->InsertOp(operation);

  return operation;
}
}  // namespace

OpTranslator::OpTranslator() : general_handler(GeneralOpHandler) {
  special_handlers["feed"] = FeedOpHandler;
  special_handlers["fetch_v2"] = FetchOpHandler;
}

}  // namespace translator
}  // namespace paddle
