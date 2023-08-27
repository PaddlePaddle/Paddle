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

#pragma once

#include <tuple>
#include <unordered_map>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"

namespace paddle {
namespace translator {

/// @brief This class is used to translate a OpDesc, it's a functor class and
/// should have no non-static data member, since we expected it's stateless.
struct OpTranscriber {
 public:
  virtual ~OpTranscriber() = default;

 public:
  using IdxInOp = size_t;
  using IdxInVector = size_t;
  using ResultIdx = std::tuple<IdxInOp, IdxInVector>;
  using OpDesc = paddle::framework::OpDesc;
  using OpOutputTypeList = std::vector<ir::Type>;
  using OpOutputMapping = std::unordered_map<std::string, ResultIdx>;
  using OpInputInfo = dialect::OpInputInfo;
  using OpInputInfoList = std::vector<dialect::OpInputInfo>;
  using OpAttributeInfo = dialect::OpAttributeInfo;
  using OpAttributeInfoList = std::vector<dialect::OpAttributeInfo>;
  using OpOutputInfo = dialect::OpOutputInfo;
  using OpOutputInfoList = std::vector<dialect::OpOutputInfo>;
  using InputHandlerFn = std::function<ir::OpResult(ir::IrContext*,
                                                    TranslationContext*,
                                                    const OpDesc&,
                                                    const std::string&,
                                                    const OpInputInfo&,
                                                    ir::Program*)>;
  using AttributeHandlerFn = std::function<ir::Attribute(
      ir::IrContext*, const OpDesc&, const OpAttributeInfo&)>;

 public:
  virtual ir::Operation* operator()(ir::IrContext* ctx,
                                    TranslationContext* param_map,
                                    const OpDesc& op_desc,
                                    ir::Program* program);

 public:
  virtual ir::OpInfo LoopkUpOpInfo(ir::IrContext* ctx, const OpDesc& op_desc);
  virtual std::vector<ir::OpResult> GenerateOperationInput(
      ir::IrContext* ctx,
      TranslationContext* param_map,
      const OpDesc& op_desc,
      const std::string& normalized_op_name,
      const OpInputInfoList& input_infos,
      ir::Program* program);
  virtual std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
      ir::IrContext* ctx,
      const OpDesc& op_desc,
      const OpOutputInfoList& output_infos);
  virtual void HandleNonexistentAttribute(ir::IrContext*,
                                          ir::AttributeMap* attribute_map,
                                          const OpAttributeInfo& info);
  virtual ir::AttributeMap TranslateOpAttribute(
      ir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc);

  virtual void RecordOpResultMapping(ir::IrContext* ctx,
                                     TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     ir::Operation* operation,
                                     const OpOutputMapping& arg_to_idx);

 public:
  virtual InputHandlerFn GetSpecialInputHandlers(
      const std::string& input_name) {
    return nullptr;
  }
  virtual AttributeHandlerFn GetSpecialAttributeHandlers(
      const std::string& input_name) {
    return nullptr;
  }
  virtual void InsertSliceOperationForInput(ir::IrContext* ctx,
                                            TranslationContext* param_map,
                                            const OpDesc& op_desc,
                                            const OpInputInfoList& input_infos,
                                            ir::Program* program);
};

class OpTranslator {
 public:
  using ResultIdx = size_t;
  using OpDesc = paddle::framework::OpDesc;
  using BlockDesc = paddle::framework::BlockDesc;
  using VarDesc = paddle::framework::VarDesc;
  using OpTranslateFn = std::function<ir::Operation*(
      ir::IrContext*, TranslationContext*, const OpDesc&, ir::Program*)>;

 private:
  OpTranslator();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, OpTranslateFn> special_handlers;
  OpTranslateFn general_handler;

 public:
  OpTranslator(const OpTranslator&) = delete;
  OpTranslator& operator=(const OpTranslator&) = delete;
  OpTranslator(OpTranslator&&) = delete;
  OpTranslator& operator=(OpTranslator&&) = delete;

  static auto& instance() {
    static OpTranslator OpTranslator;
    return OpTranslator;
  }

  OpTranslateFn& operator[](const std::string& op_type) {
    if (special_handlers.count(op_type) == 0) {
      return general_handler;
    } else {
      return special_handlers[op_type];
    }
  }

  bool HasSpecialHandler(const std::string& op_type) {
    return special_handlers.count(op_type) != 0;
  }
};

using OpTranslateFn = OpTranslator::OpTranslateFn;

}  // namespace translator
}  // namespace paddle
