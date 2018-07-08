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

#include <algorithm>
#include <fstream>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/pybind/pybind.h"

DEFINE_string(devices, "", "The devices to be used which is joined by comma.");
DEFINE_bool(init_p2p, false, "Whether to init p2p.");
DEFINE_int32(math_num_threads, 1,
             "Number of threads used to run math functions.");

namespace paddle {
namespace inference {

void Init(const std::vector<std::string> argv) {
  framework::InitGflags(argv);
  operators::math::SetNumThreads(FLAGS_math_num_threads);
  // init devices
  std::vector<int> devices;
  std::string token;
  std::istringstream tokenStream(FLAGS_devices);
  while (std::getline(tokenStream, token, ',')) {
    devices.push_back(std::stoi(token));
  }
  framework::InitDevices(FLAGS_init_p2p, devices);
}

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

bool IsPersistable(const framework::VarDesc* var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST) {
    return true;
  }
  return false;
}

void LoadPersistables(framework::Executor* executor, framework::Scope* scope,
                      const framework::ProgramDesc& main_program,
                      const std::string& dirname,
                      const std::string& param_filename) {
  const framework::BlockDesc& global_block = main_program.Block(0);

  framework::ProgramDesc* load_program = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program->MutableBlock(0);
  std::vector<std::string> paramlist;

  for (auto* var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();

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

  executor->Run(*load_program, scope, 0, true, true);

  delete load_program;
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& dirname) {
  std::string model_filename = dirname + "/__model__";
  std::string program_desc_str;
  VLOG(3) << "loading model from " << model_filename;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  LoadPersistables(executor, scope, *main_program, dirname, "");
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> Load(
    framework::Executor* executor, framework::Scope* scope,
    const std::string& prog_filename, const std::string& param_filename) {
  std::string model_filename = prog_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));

  LoadPersistables(executor, scope, *main_program, "", param_filename);
  return main_program;
}

}  // namespace inference
}  // namespace paddle
