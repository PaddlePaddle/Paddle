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

static bool IsNumber(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

static std::string GenePath(const std::string &dir, const std::string &file) {
  std::string file_path;
  file_path.append(dir);
  file_path.append("/");
  file_path.append(file);
  return file_path;
}

static bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static void MkDir(const char *path) {
  if (mkdir(path, 0755)) {
    PADDLE_ENFORCE_EQ(errno, EEXIST, "%s mkdir failed!", path);
  }
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}

class CheckpointSaveOp : public framework::OperatorBase {
 public:
  CheckpointSaveOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto ck_dir = Attr<std::string>("dir");
    auto overwrite = Attr<bool>("overwrite");

    std::string serial_var_name = std::string(SERIAL_VAR);
    auto *serial_num =
        scope.FindVar(serial_var_name)->GetMutable<std::string>();
    VLOG(1) << "CheckpointSaveOp get " << SERIAL_VAR
            << " value: " << serial_num;

    if (serial_num->empty()) {
      serial_num->append("0");
    }

    std::string dir = GenePath(ck_dir, serial_num->c_str());
    VLOG(1) << "CheckpointSaveOp current dir: " << dir;
    bool is_present = FileExists(dir);
    if (is_present && !overwrite) {
      PADDLE_THROW("%s exists!, checkpoint save cannot to overwrite it", dir,
                   overwrite);
    }
    MkDirRecursively(dir.c_str());

    auto inp_var_names = Inputs("X");
    PADDLE_ENFORCE_GT(static_cast<int>(inp_var_names.size()), 0,
                      "The number of input variables should be greater than 0");

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    // todo (tangwei) made it async
    for (size_t i = 0; i < inp_var_names.size(); i++) {
      auto *var = scope.FindVar(inp_var_names[i]);

      PADDLE_ENFORCE(var != nullptr,
                     "Cannot find variable %s for checkpoint save op",
                     inp_var_names[i]);
      PADDLE_ENFORCE(
          var->IsType<framework::LoDTensor>(),
          "CheckpointSaveOp only supports LoDTensor, %s has wrong type",
          inp_var_names[i]);

      auto &tensor = var->Get<framework::LoDTensor>();
      // Serialize tensors one by one
      std::string var_file = GenePath(dir, inp_var_names[i]);
      std::ofstream fout(var_file);
      framework::SerializeToStream(fout, tensor, dev_ctx);
      fout.close();
    }

    std::string success = GenePath(dir, SUCCESS);
    std::ofstream fout(success);
    fout.close();
  }
};

class CheckpointSaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CheckpointSaveOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(vector) Input LoDTensors that need to be saved together in a file.")
        .AsDuplicable();
    AddComment(R"DOC(
CheckpointSave operator

This operator will serialize and write a list of input LoDTensor variables
to a file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default false)"
                  "Delete the output dir if it exists.")
        .SetDefault(false);

    AddAttr<std::string>("dir",
                         "(string)"
                         "The dir where the LoDTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

// class CheckpointSaveOpVarTypeInference : public framework::VarTypeInference {
//  public:
//   void operator()(const framework::OpDesc &op_desc,
//                   framework::BlockDesc *block) const override {
//     auto out_var_name = op_desc.Output("Serial").front();
//     auto &out_var = block->FindRecursiveOrCreateVar(out_var_name);
//     auto var_type = framework::proto::VarType::RAW;
//     out_var.SetType(var_type);
//   }
// };

// class CheckpointSaveOpShapeInference : public framework::InferShapeBase {
//  public:
//   void operator()(framework::InferShapeContext *ctx) const override {}
// };

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(checkpoint_save, ops::CheckpointSaveOp,
                  ops::CheckpointSaveOpProtoMaker);

// REGISTER_OPERATOR(checkpoint_save, ops::CheckpointSaveOp,
//                   paddle::framework::EmptyGradOpMaker,
//                   ops::CheckpointSaveOpProtoMaker,
//                   ops::CheckpointSaveOpVarTypeInference,
//                   ops::CheckpointSaveOpShapeInference);
