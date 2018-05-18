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

#include <stdint.h>
#include <sys/stat.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <string>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

constexpr char kSEP = '/';
// write empty file named _SUCCESS
const char SUCCESS[] = "_SUCCESS";
const char SERIAL_VAR[] = "SERIAL_NUMBER";

static bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

static std::string GenePath(const std::string &dir, const std::string &file) {
  std::string file_path;
  file_path.append(file_path);
  file_path.append("/");
  file_path.append(file);
  return file_path;
}

static bool IsNumber(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

static void LoadInputVars(const framework::Scope &scope,
                          const platform::Place &place,
                          const std::vector<std::string> &inp_var_names,
                          const std::string &dir) {
  // get device context from pool
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  // todo (tangwei) made it async
  for (size_t i = 0; i < inp_var_names.size(); i++) {
    auto *var = scope.FindVar(inp_var_names[i]);

    PADDLE_ENFORCE(var != nullptr,
                   "Cannot find variable %s for save_combine_op",
                   inp_var_names[i]);
    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "LoadCombineOp only supports LoDTensor, %s has wrong type",
                   inp_var_names[i]);

    std::string var_file = GenePath(dir, inp_var_names[i]);
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    std::ifstream fin(var_file);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   var_file);
    framework::DeserializeFromStream(fin, tensor, dev_ctx);
    fin.close();
    VLOG(3) << " load var: " << inp_var_names[i] << " finished";
  }
}

static void LoadStringArgv(const framework::Scope &scope,
                           const platform::Place &place,
                           const std::vector<std::string> &argv,
                           const std::string &dir) {
  for (size_t i = 0; i < argv.size(); i++) {
    auto *var = scope.FindVar(argv[i]);
    std::string *var_str = var->GetMutable<std::string>();
    std::string var_file = GenePath(dir, argv[i]);
    std::ifstream fin(var_file);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   var_file);
    std::getline(fin, *var_str);
    fin.close();
    VLOG(3) << " load String argv: " << argv[i] << " value is: " << var_str;
  }
}

class CheckpointLoadOp : public framework::OperatorBase {
 public:
  CheckpointLoadOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    std::string dir = Attr<std::string>("dir");
    std::string serial_num_attr = Attr<std::string>("Serial");

    PADDLE_ENFORCE(!IsNumber(serial_num_attr),
                   "Checkpoint Serial must be a number");

    std::string serial_var_name = std::string(SERIAL_VAR);
    auto *serial_var = scope.FindVar(serial_var_name);
    PADDLE_ENFORCE(serial_var != nullptr,
                   "Cannot find variable %s for checkpoint_load_op",
                   serial_var_name);

    auto *serial_num = serial_var->GetMutable<std::string>();
    serial_num->clear();
    serial_num->append(serial_num_attr);

    VLOG(1) << "CheckpointLoadOp set " << SERIAL_VAR
            << " value: " << serial_num;

    std::string success = GenePath(dir, serial_num->c_str());
    VLOG(3) << "Load checkpoint from dir: " << success;
    success = GenePath(success, SUCCESS);
    bool is_present = FileExists(success);
    if (!is_present) {
      VLOG(1) << "CheckpointLoadOp can not find " << SUCCESS
              << " from: " << success;
      return;
    }

    VLOG(3) << "Ready to load vars to scope";
    auto inp_var_names = Inputs("X");
    PADDLE_ENFORCE_GT(static_cast<int>(inp_var_names.size()), 0,
                      "The number of input variables should be greater than 0");
    LoadInputVars(scope, place, inp_var_names, dir);

    // VLOG(3) << "Ready to load string argv to scope";
    // auto argv = Output("Argv");
    // LoadStringArgv(scope, place, argv, dir);
  }
};

class CheckpointLoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(vector) Input LoDTensors that need to be saved together in a file.")
        .AsDuplicable();
    AddOutput(
        "Argv",
        "(vector) Input LoDTensors that need to be saved together in a file.");
    AddComment(R"DOC(
CheckpointLoad operator

This operator will serialize and write a list of input LoDTensor variables
to a file on disk.
)DOC");

    AddAttr<std::string>(
        "Serial",
        "(std::string)"
        "The  serial number of the checkpoint will to be load.");
    AddAttr<std::string>(
        "dir",
        "(string)"
        "The \"file_path\" where the LoDTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

class CheckpointLoadOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto out_var_name = op_desc.Output("Argv").front();
    auto &out_var = block->FindRecursiveOrCreateVar(out_var_name);
    auto var_type = framework::proto::VarType::RAW;
    out_var.SetType(var_type);
  }
};

class CheckpointLoadOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(checkpoint_load, ops::CheckpointLoadOp,
                  paddle::framework::EmptyGradOpMaker,
                  ops::CheckpointLoadOpProtoMaker,
                  ops::CheckpointLoadOpVarTypeInference,
                  ops::CheckpointLoadOpShapeInference);

// REGISTER_OPERATOR(checkpoint_load, ops::CheckpointLoadOp,
//                  ops::CheckpointLoadOpProtoMaker);
