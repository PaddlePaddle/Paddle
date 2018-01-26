/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "inference.h"
#include <fstream>
#include "paddle/framework/executor.h"
#include "paddle/framework/feed_fetch_method.h"
#include "paddle/framework/scope.h"

namespace paddle {

namespace infer {

bool IsParameter(const framework::VarDesc* var,
                 const framework::ProgramDesc* main_program) {
  if (var->Persistable()) {
    // There are many unreachable variables in the program
    for (size_t i = 0; i < main_program->Size(); ++i) {
      const framework::BlockDesc& block = main_program->Block(i);
      for (auto* op : block.AllOps()) {
        if (op->Type() == "feed") {
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
                      framework::ProgramDesc* main_program) {
  framework::BlockDesc* global_block = main_program->MutableBlock(0);

  framework::ProgramDesc* load_program = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program->MutableBlock(0);
  for (auto* var : global_block->AllVars()) {
    if (IsParameter(var, main_program)) {
      LOG(INFO) << "parameter's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->Shape());
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
}

framework::ProgramDesc* LoadModelAndParam(framework::Executor& executor,
                                          framework::Scope& scope,
                                          const std::string& dirname) {
  std::string model_filename = dirname + "/__model__.dat";
  LOG(INFO) << "loading model from " << model_filename;
  std::ifstream inputfs(model_filename, std::ios::in | std::ios::binary);
  std::string program_desc_str;
  inputfs.seekg(0, std::ios::end);
  program_desc_str.resize(inputfs.tellg());
  inputfs.seekg(0, std::ios::beg);
  LOG(INFO) << "program_desc_str's size: " << program_desc_str.size();
  inputfs.read(&program_desc_str[0], program_desc_str.size());
  inputfs.close();

  framework::ProgramDesc* main_program =
      new framework::ProgramDesc(program_desc_str);

  LoadPersistables(executor, scope, dirname, main_program);
  framework::BlockDesc* global_block = main_program->MutableBlock(0);

  for (auto* op : global_block->AllOps()) {
    if (op->Type() == "feed") {
      main_program->InsertFeedVarName(op->Output("Out")[0]);
    } else if (op->Type() == "fetch") {
      main_program->InsertFetchVarName(op->Input("X")[0]);
    }
  }
  return main_program;
}
}  // namespace infer

void InferenceEngine::LoadInferenceModel(framework::Executor& executor,
                                         framework::Scope& scope,
                                         const std::string& dirname) {
  program_ = infer::LoadModelAndParam(executor, scope, dirname);
}

void InferenceEngine::PrependFeedOp() {
  if (!program_) {
    LOG(FATAL) << "Please initialize the program_ first.";
  }

  framework::BlockDesc* global_block = program_->MutableBlock(0);

  // create_var
  framework::VarDesc* feed_var = global_block->Var("feed");
  feed_var->SetType(framework::proto::VarDesc::FEED_MINIBATCH);
  feed_var->SetPersistable(true);

  // prepend feed_op
  const std::vector<std::string>& feed_var_names = program_->GetFeedVarNames();
  for (size_t i = 0; i < feed_var_names.size(); ++i) {
    std::string var_name = feed_var_names[i];
    LOG(INFO) << "feed var's name: " << var_name;

    // prepend_op
    framework::OpDesc* op = global_block->PrependOp();
    op->SetType("feed");
    op->SetInput("X", {"feed"});
    op->SetOutput("Out", {var_name});
    op->SetAttr("col", {static_cast<int>(i)});
    op->CheckAttrs();
  }
}

void InferenceEngine::AppendFetchOp() {
  if (!program_) {
    LOG(FATAL) << "Please initialize the program_ first.";
  }

  framework::BlockDesc* global_block = program_->MutableBlock(0);

  // create_var
  framework::VarDesc* fetch_var = global_block->Var("fetch");
  fetch_var->SetType(framework::proto::VarDesc::FETCH_LIST);
  fetch_var->SetPersistable(true);

  // append fetch_op
  const std::vector<std::string>& fetch_var_names =
      program_->GetFetchVarNames();
  for (size_t i = 0; i < fetch_var_names.size(); ++i) {
    std::string var_name = fetch_var_names[i];
    LOG(INFO) << "fetch var's name: " << var_name;

    // append_op
    framework::OpDesc* op = global_block->AppendOp();
    op->SetType("fetch");
    op->SetInput("X", {var_name});
    op->SetOutput("Out", {"fetch"});
    op->SetAttr("col", {static_cast<int>(i)});
    op->CheckAttrs();
  }
}

void InferenceEngine::Execute(framework::Executor* executor,
                              framework::Scope* scope,
                              const std::vector<framework::LoDTensor>& feeds,
                              std::vector<framework::LoDTensor>& fetchs) {
  if (!program_) {
    LOG(FATAL) << "Please initialize the program_ first.";
  }

  const std::vector<std::string> feed_var_names = program_->GetFeedVarNames();
  if (feeds.size() < feed_var_names.size()) {
    LOG(FATAL) << "Please feed " << feed_var_names.size() << " input Tensors.";
  }

  // set_feed_variable
  for (size_t i = 0; i < feed_var_names.size(); ++i) {
    framework::SetFeedVariable(scope, feeds[i], "feed", i);
  }

  executor->Run(*program_, scope, 0, true, true);

  // get_fetch_variable
  const std::vector<std::string> fetch_var_names = program_->GetFetchVarNames();
  fetchs.resize(fetch_var_names.size());
  for (size_t i = 0; i < fetch_var_names.size(); ++i) {
    fetchs[i] = framework::GetFetchVariable(*scope, "fetch", i);
  }
}
}  // namespace paddle
