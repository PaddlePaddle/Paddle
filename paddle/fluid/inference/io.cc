/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/io.h"

#include <fstream>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"

namespace paddle {
namespace inference {

void ReadBinaryFile(const std::string& filename, std::string& contents) {
  VLOG(3) << "loading model from " << filename;
  std::ifstream inputfs(filename, std::ios::in | std::ios::binary);
  inputfs.seekg(0, std::ios::end);
  contents.clear();
  contents.resize(inputfs.tellg());
  inputfs.seekg(0, std::ios::beg);
  inputfs.read(&contents[0], contents.size());
  inputfs.close();
}

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
                      const framework::ProgramDesc& main_program,
                      const std::string& dirname,
                      const std::string& param_filename) {
  const framework::BlockDesc& global_block = main_program.Block(0);

  framework::ProgramDesc* load_program = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program->MutableBlock(0);
  std::vector<std::string> paramlist;

  for (auto* var : global_block.AllVars()) {
    if (IsParameter(var, main_program)) {
      VLOG(3) << "parameter's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      if (!param_filename.empty()) {
        paramlist.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc* op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {dirname + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!param_filename.empty()) {
    // sort paramlist to have consistent ordering
    std::sort(paramlist.begin(), paramlist.end());
    // append just the load_combine op
    framework::OpDesc* op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", paramlist);
    op->SetAttr("file_path", {param_filename});
    op->CheckAttrs();
  }

  executor.Run(*load_program, &scope, 0, true, true);

  VLOG(3) << "Ran loading successfully";
  delete load_program;
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor& executor,
                                             framework::Scope& scope,
                                             const std::string& dirname) {
  std::string model_filename = dirname + "/__model__";
  std::string program_desc_str;
  ReadBinaryFile(model_filename, program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  LoadPersistables(executor, scope, *main_program, dirname, "");
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> Load(
    framework::Executor& executor,
    framework::Scope& scope,
    const std::string& prog_filename,
    const std::string& param_filename) {
  std::string model_filename = prog_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  LoadPersistables(executor, scope, *main_program, "", param_filename);
  return main_program;
}

}  // namespace inference
}  // namespace paddle
