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

#include <stdint.h>
#include <sys/stat.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {

// TODO(sidgoyal78): These function are needed by other files (save_op), move
// them to paddle::filesystem namespace. (as noted by yuyang18 in save_op).
constexpr char kSEP = '/';
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

class SaveCombineOp : public framework::OperatorBase {
 public:
  SaveCombineOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    auto filename = Attr<std::string>("file_path");
    auto overwrite = Attr<bool>("overwrite");

    bool is_present = FileExists(filename);
    if (is_present && !overwrite) {
      PADDLE_THROW("%s exists!, cannot save_combine to it when overwrite=false",
                   filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());
    std::ofstream fout(filename);
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

    auto inp_var_names = Inputs("X");
    PADDLE_ENFORCE_GT(static_cast<int>(inp_var_names.size()), 0,
                      "The number of input variables should be greater than 0");

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    for (size_t i = 0; i < inp_var_names.size(); i++) {
      auto *var = scope.FindVar(inp_var_names[i]);

      PADDLE_ENFORCE(var != nullptr,
                     "Cannot find variable %s for save_combine_op",
                     inp_var_names[i]);
      PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                     "SaveCombineOp only supports LoDTensor, %s has wrong type",
                     inp_var_names[i]);

      auto &tensor = var->Get<framework::LoDTensor>();
      // Serialize tensor
      framework::SerializeToStream(fout, tensor, dev_ctx);
    }
    fout.close();
  }
};

class SaveCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveCombineOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(vector) Input LoDTensors that need to be saved together in a file.")
        .AsDuplicable();
    AddComment(R"DOC(
SaveCombine operator

This operator will serialize and write a list of input LoDTensor variables 
to a file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if it exists.")
        .SetDefault(true);
    AddAttr<std::string>(
        "file_path",
        "(string)"
        "The \"file_path\" where the LoDTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save_combine, ops::SaveCombineOp,
                  ops::SaveCombineOpProtoMaker);
