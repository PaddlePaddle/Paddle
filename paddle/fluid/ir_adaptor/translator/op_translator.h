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
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

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
  using OpOutputTypeList = std::vector<pir::Type>;
  using OpOutputMapping = std::unordered_map<std::string, ResultIdx>;
  using OpInputInfo = dialect::OpInputInfo;
  using OpInputInfoList = std::vector<dialect::OpInputInfo>;
  using OpAttributeInfo = dialect::OpAttributeInfo;
  using OpAttributeInfoList = std::vector<dialect::OpAttributeInfo>;
  using OpOutputInfo = dialect::OpOutputInfo;
  using OpOutputInfoList = std::vector<dialect::OpOutputInfo>;
  using InputHandlerFn = std::function<pir::Value(pir::IrContext*,
                                                  TranslationContext*,
                                                  const OpDesc&,
                                                  const std::string&,
                                                  const OpInputInfo&,
                                                  pir::Block*)>;
  using AttributeHandlerFn = std::function<pir::Attribute(
      pir::IrContext*, const OpDesc&, const OpAttributeInfo&)>;

 public:
  virtual pir::Operation* operator()(pir::IrContext* ctx,
                                     TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     pir::Block* block);

 public:
  virtual pir::OpInfo LookUpOpInfo(pir::IrContext* ctx, const OpDesc& op_desc);
  virtual std::vector<pir::Value> GenerateOperationInput(
      pir::IrContext* ctx,
      TranslationContext* param_map,
      const OpDesc& op_desc,
      const std::string& normalized_op_name,
      const OpInputInfoList& input_infos,
      pir::Block* block);
  virtual std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
      pir::IrContext* ctx,
      const OpDesc& op_desc,
      const OpOutputInfoList& output_infos);
  virtual void HandleNonexistentAttribute(pir::IrContext*,
                                          pir::AttributeMap* attribute_map,
                                          const OpAttributeInfo& info);
  virtual pir::AttributeMap TranslateOpAttribute(
      pir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc);
  virtual pir::Value GetAttributeAsInput(pir::IrContext* ctx,
                                         pir::Block* block,
                                         const OpDesc& op_desc,
                                         const OpInputInfo& input_info);

  virtual void RecordOpResultMapping(pir::IrContext* ctx,
                                     TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     pir::Operation* operation,
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
  virtual void InsertSliceOperationForInput(pir::IrContext* ctx,
                                            TranslationContext* param_map,
                                            const OpDesc& op_desc,
                                            const OpInputInfoList& input_infos,
                                            pir::Block* block);
};

class OpTranslator {
 public:
  using ResultIdx = size_t;
  using OpDesc = paddle::framework::OpDesc;
  using BlockDesc = paddle::framework::BlockDesc;
  using VarDesc = paddle::framework::VarDesc;
  using OpTranslateFn = std::function<pir::Operation*(
      pir::IrContext*, TranslationContext*, const OpDesc&, pir::Block*)>;

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
