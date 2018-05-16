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

static bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
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

    VLOG(3) << "Load checkpoint from dir: " << dir;

    std::string success;
    success.append(dir);
    success.append("/");
    success.append(SUCCESS);

    bool is_present = FileExists(success);
    if (!is_present) {
      VLOG(3) << "can not find _SUCCESS from  path: " << success;
      return;
    }

    auto inp_var_names = Output("Out");
    PADDLE_ENFORCE_GT(static_cast<int>(inp_var_names.size()), 0,
                      "The number of input variables should be greater than 0");
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
                     "SaveCombineOp only supports LoDTensor, %s has wrong type",
                     inp_var_names[i]);

      std::string var_file;
      var_file.append(dir);
      var_file.append("/");
      var_file.append(inp_var_names[i]);
      VLOG(3) << "ready to load var: " << inp_var_names[i];

      auto &tensor = var->Get<framework::LoDTensor>();

      std::ifstream fin(var_file);
      PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                     var_file);
      DeserializeFromStream(fin, tensor, *dev_ctx);
      fin.close();
      VLOG(3) << " load var: " << inp_var_names[i] << " finished";
    }
  }
};

class CheckpointLoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CheckpointLoadOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "(Tensor) The tensor need to be loaded");
    AddComment(R"DOC(
CheckpointLoad operator

This operator will serialize and write a list of input LoDTensor variables 
to a file on disk.
)DOC");

    AddAttr<std::string>(
        "dir",
        "(string)"
        "The \"file_path\" where the LoDTensor variables will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(checkpoint_load, ops::CheckpointLoadOp,
                  ops::CheckpointLoadOpProtoMaker);
