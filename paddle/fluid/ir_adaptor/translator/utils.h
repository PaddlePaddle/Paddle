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

#include <string>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"

namespace paddle {
namespace dialect {
struct PdOpSig {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  PdOpSig() = default;
  PdOpSig(const PdOpSig& input_info) = default;

  PdOpSig(const std::string& name,
          const std::vector<std::string>& inputs,
          const std::vector<std::string>& outputs)
      : name(name), inputs(inputs), outputs(outputs) {}
};

bool HaveOpToMultiKernelsMap(std::string op_name);

const std::vector<PdOpSig>& LegacyOpToPdOpsMapping(std::string op_name);

}  // namespace dialect
}  // namespace paddle

namespace paddle {
namespace translator {

pir::Operation* InsertSliceOperationForTarget(
    pir::IrContext* ctx,
    TranslationContext* param_map,
    pir::Block* block,
    const VariableDefiningInfo& defining_info,
    const std::string& arg_name);

std::ostream& operator<<(std::ostream& os,
                         const std::vector<std::string>& vec_str);

std::vector<std::string> CheckUnregisteredOperation(
    pir::IrContext* ctx, const framework::ProgramDesc& legacy_program);

}  // namespace translator
}  // namespace paddle
