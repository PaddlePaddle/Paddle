/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/inference/io.h"

#include <fstream>
#include "paddle/framework/block_desc.h"
#include "paddle/framework/feed_fetch_type.h"

namespace paddle {
namespace inference {

bool IsParameter(const framework::VarDesc* var,
                 const framework::ProgramDesc& main_program) {
  if (var->Persistable()) {
    // There are many unreachable variables in the program
    for (size_t i = 0; i < main_program.Size(); ++i) {
      const framework::BlockDesc& block = main_program.Block(i);
      for (auto* op : block.AllOps()) {
        if (op->Type() == framework::kFeedOpType) {
          continue;
        }
        for (auto input_argument_name : op->InputArgumentNames()) {
          if (input_argument_name == var->Name()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void LoadPersistables(framework::Executor& executor,
                      framework::Scope& scope,
                      const std::string& dirname,
                      const framework::ProgramDesc& main_program) {
  const framework::BlockDesc& global_block = main_program.Block(0);

  framework::ProgramDesc* load_program = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program->MutableBlock(0);
  for (auto* var : global_block.AllVars()) {
    if (IsParameter(var, main_program)) {
      VLOG(3) << "parameter's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      // append_op
      framework::OpDesc* op = load_block->AppendOp();
      op->SetType("load");
      op->SetOutput("Out", {new_var->Name()});
      op->SetAttr("file_path", {dirname + "/" + new_var->Name()});
      op->CheckAttrs();
    }
  }
  executor.Run(*load_program, &scope, 0, true, true);
  delete load_program;
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor& executor,
                                             framework::Scope& scope,
                                             const std::string& dirname) {
  std::string model_filename = dirname + "/__model__";
  LOG(INFO) << "loading model from " << model_filename;
  std::ifstream inputfs(model_filename, std::ios::in | std::ios::binary);
  std::string program_desc_str;
  inputfs.seekg(0, std::ios::end);
  program_desc_str.resize(inputfs.tellg());
  inputfs.seekg(0, std::ios::beg);
  LOG(INFO) << "program_desc_str's size: " << program_desc_str.size();
  inputfs.read(&program_desc_str[0], program_desc_str.size());
  inputfs.close();

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  LoadPersistables(executor, scope, dirname, *main_program);
  return main_program;
}

}  // namespace inference
}  // namespace paddle
