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
#include "paddle/framework/init.h"
#include "paddle/framework/scope.h"

#ifdef PADDLE_USE_PTOOLS
#include "chooseser.h"
#endif

namespace paddle {

void InferenceEngine::LoadInferenceModel(
    const std::string& dirname,
    const std::vector<std::string>& feed_var_names,
    const std::vector<std::string>& fetch_var_names) {
#ifdef PADDLE_USE_PTOOLS
  std::string model_filename = dirname + "/__model__";
  LOG(INFO) << "Using PicklingTools, loading model from " << model_filename;
  Val v;
  LoadValFromFile(model_filename.c_str(), v, SERIALIZE_P0);
  std::string program_desc_str = v["program_desc_str"];
  LOG(INFO) << "program_desc_str's size: " << program_desc_str.size();
// PicklingTools cannot parse the vector of strings correctly.
#else
  // program_desc_str
  // the inference.model is stored by following python codes:
  //  inference_program = fluid.io.get_inference_program(predict)
  //  model_filename = "recognize_digits_mlp.inference.model/inference.model"
  //  with open(model_filename, "w") as f:
  //      program_str = inference_program.desc.serialize_to_string()
  //          f.write(struct.pack('q', len(program_str)))
  //          f.write(program_str)
  std::string model_filename = dirname + "/inference.model";
  LOG(INFO) << "loading model from " << model_filename;
  std::ifstream fs(model_filename, std::ios_base::binary);
  int64_t size = 0;
  fs.read(reinterpret_cast<char*>(&size), sizeof(int64_t));
  LOG(INFO) << "program_desc_str's size: " << size;
  std::string program_desc_str;
  program_desc_str.resize(size);
  fs.read(&program_desc_str[0], size);
#endif
  program_ = new framework::ProgramDesc(program_desc_str);
  GenerateLoadProgram(dirname);

  if (feed_var_names.empty() || fetch_var_names.empty()) {
    LOG(FATAL) << "Please specify the feed_var_names and fetch_var_names.";
  }
  feed_var_names_ = feed_var_names;
  fetch_var_names_ = fetch_var_names;
  PrependFeedOp();
  AppendFetchOp();
}

bool InferenceEngine::IsParameter(const framework::VarDesc* var) {
  if (var->Persistable()) {
    // There are many unreachable variables in the program
    for (size_t i = 0; i < program_->Size(); ++i) {
      const framework::BlockDesc& block = program_->Block(i);
      for (auto* op : block.AllOps()) {
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

void InferenceEngine::GenerateLoadProgram(const std::string& dirname) {
  framework::BlockDesc* global_block = program_->MutableBlock(0);

  load_program_ = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program_->MutableBlock(0);
  for (auto* var : global_block->AllVars()) {
    if (IsParameter(var)) {
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
  for (size_t i = 0; i < feed_var_names_.size(); ++i) {
    std::string var_name = feed_var_names_[i];
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
  for (size_t i = 0; i < fetch_var_names_.size(); ++i) {
    std::string var_name = fetch_var_names_[i];
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

void InferenceEngine::Execute(const std::vector<framework::LoDTensor>& feeds,
                              std::vector<framework::LoDTensor>& fetchs) {
  if (!program_ || !load_program_) {
    LOG(FATAL) << "Please initialize the program_ and load_program_ first.";
  }

  if (feeds.size() < feed_var_names_.size()) {
    LOG(FATAL) << "Please feed " << feed_var_names_.size() << " input Tensors.";
  }

  auto* place = new platform::CPUPlace();
  framework::InitDevices({"CPU"});
  framework::Executor* executor = new framework::Executor(*place);
  framework::Scope* scope = new framework::Scope();

  executor->Run(*load_program_, scope, 0, true, true);

  // set_feed_variable
  for (size_t i = 0; i < feed_var_names_.size(); ++i) {
    framework::SetFeedVariable(scope, feeds[i], "feed", i);
  }

  executor->Run(*program_, scope, 0, true, true);

  // get_fetch_variable
  fetchs.resize(fetch_var_names_.size());
  for (size_t i = 0; i < fetch_var_names_.size(); ++i) {
    fetchs[i] = framework::GetFetchVariable(*scope, "fetch", i);
  }

  delete place;
  delete scope;
  delete executor;
}
}  // namespace paddle
