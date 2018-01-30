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
    auto position_counter = Attr<int>("position_counter");

    bool is_present = FileExists(filename);
    if (is_present && !overwrite && position_counter == 0) {
      PADDLE_THROW(
          "%s is existed, cannot save_combine to it when overwrite=false",
          filename, overwrite);
    }

    MkDirRecursively(DirName(filename).c_str());

    std::ofstream fout;

    // if position_counter is 0, we open the file in write mode,
    // otherwise, we open in append mode.
    if (position_counter == 0) {
      fout.open(filename);
    } else {
      fout.open(filename, std::ios_base::app);
    }
    PADDLE_ENFORCE(static_cast<bool>(fout), "Cannot open %s to write",
                   filename);

    auto iname = Input("X");

    auto *var = scope.FindVar(iname);
    PADDLE_ENFORCE(var != nullptr,
                   "Cannot find variable %s for save_combine_op", iname);

    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "SaveCombineOp only support LoDTensor, %s has wrong type",
                   iname);

    auto &tensor = var->Get<framework::LoDTensor>();

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    // Create "output string stream" to get the serialized LodTensor
    std::ostringstream str_stream;
    framework::SerializeToStream(str_stream, tensor, dev_ctx);
    std::string current_serialized_data = str_stream.str();

    // Save 'current_size' information as a fixed width integer, and
    // further save the serialized data using 'current_size' bytes
    uint64_t current_size = current_serialized_data.size();
    fout.write(reinterpret_cast<const char *>(&current_size),
               sizeof(current_size));
    fout.write(reinterpret_cast<const char *>(current_serialized_data.c_str()),
               static_cast<std::streamsize>(current_size));
    fout.close();
  }
};

class SaveCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveCombineOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor ) Input tensor to be save_combined");
    AddComment(R"DOC(
Save_combine operator

This operator will serialize and write tensor variables to file on disk in a 
combined fashion.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<int>("position_counter",
                 "(int) "
                 "It specifies the relative ordering of different parameters.")
        .AddCustomChecker([](const int &counter) { return counter >= 0; });
    AddAttr<std::string>(
        "file_path",
        "(string)"
        "The \"file_path\" where the variable will be save_combined.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save_combine, ops::SaveCombineOp,
                  ops::SaveCombineOpProtoMaker);
