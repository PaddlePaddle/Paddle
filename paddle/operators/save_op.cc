/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
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

class SaveOp : public framework::OperatorBase {
 public:
  SaveOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto filename = Attr<std::string>("file_path");
    auto overwrite = Attr<bool>("overwrite");

    if (FileExists(filename) && !overwrite) {
      PADDLE_THROW("%s is existed, cannot save to it when overwrite=false",
                   filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(filename);
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

    auto iname = Input("X");
    auto *var = scope.FindVar(iname);
    PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s for save_op",
                   iname);

    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "SaveOp only support LoDTensor, %s has wrong type", iname);

    auto &tensor = var->Get<framework::LoDTensor>();
    framework::SerializeToStream(fout, tensor);
  }
};

class SaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveOpProtoMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor ) Input tensor to be saved");
    AddComment(R"DOC(
Save operator

This operator will serialize and write a tensor variable to file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker);
