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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/program.h"
#include "paddle/ir/value.h"

namespace paddle {
namespace translator {

using TranslationContext = std::unordered_map<std::string, ir::OpResult>;

class ProgramTranslator {
  using ProgramDesc = ::paddle::framework::ProgramDesc;
  using BlockDesc = ::paddle::framework::BlockDesc;

 public:
  explicit ProgramTranslator(const ProgramDesc* legacy_program,
                             ir::Program* program);

  void Translate();

 private:
  const ProgramDesc* legacy_program;
  ir::Program* program;
  TranslationContext param_map;
  ir::IrContext* ctx;

  void ExtractParameterFromSingleBlock(const BlockDesc& block);
  void InsertOperationToSingleBlock(const BlockDesc& block);
};

}  // namespace translator
}  // namespace paddle
