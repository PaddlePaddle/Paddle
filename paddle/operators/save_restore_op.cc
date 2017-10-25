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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

#include <fstream>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::LoDTensor;

inline static std::string VarToFileName(const std::string& folder_path,
                                        const std::string& var_name) {
  return folder_path + "/__" + var_name + "__";
}

class SaveOp : public framework::OperatorBase {
 public:
  SaveOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    const auto& var_names = this->Inputs("X");
    for (const auto& name : var_names) {
      PADDLE_ENFORCE_NOT_NULL(scope.FindVar(name),
                              "Can not find variable '%s' in the scope.", name);
    }
    std::string folder_path = this->Attr<std::string>("folderPath");
    PADDLE_ENFORCE(!folder_path.empty(),
                   "'folderPath' of SaveOp shouldn't be empty.");

    VLOG(1) << "Save variables to folder: " << folder_path;
    for (const auto& name : var_names) {
      std::string file_name = VarToFileName(folder_path, name);
      std::ofstream fout(file_name, std::ofstream::out);
      PADDLE_ENFORCE(fout.is_open(), "Fail to create file %s.", file_name);
      const LoDTensor& tensor = scope.FindVar(name)->Get<LoDTensor>();
      std::string bytes = tensor.SerializeToString();
      fout << bytes;
      fout.close();
    }
    VLOG(1) << "Compelete saving variables. Items count: " << var_names.size();
  }
};

class SaveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SaveOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(tensor), the tensor count can be 1~INT_MAX, tensors names which "
             "values will be saved.")
        .AsDuplicable();
    AddAttr<std::string>("folderPath", "the folderPath for save model.");
    AddComment(R"DOC(
Save the input tensors to a binary file based on input tensor names and absolute path.

All the inputs can carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

class RestoreOp : public framework::OperatorBase {
 public:
  RestoreOp(const std::string& type, const framework::VariableNameMap& inputs,
            const framework::VariableNameMap& outputs,
            const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    const auto& var_names = this->Outputs("Out");
    for (const auto& name : var_names) {
      PADDLE_ENFORCE_NOT_NULL(scope.FindVar(name),
                              "Can not find variable '%s' in the scope.", name);
    }
    std::string folder_path = this->Attr<std::string>("folderPath");
    PADDLE_ENFORCE(!folder_path.empty(),
                   "'folderPath' of RestoreOp shouldn't be empty.");

    VLOG(1) << "Try loading variables from folder: " << folder_path;

    for (const auto& name : var_names) {
      std::string file_name = VarToFileName(folder_path, name);
      std::ifstream fin(file_name, std::ifstream::in);
      PADDLE_ENFORCE(fin.is_open(), "Fail to open file %s.", file_name);
      const size_t kBufferSize = 4096;  // equal to linux page size
      char buffer[kBufferSize];
      std::string cache;
      while (!fin.eof()) {
        fin.read(buffer, kBufferSize);
        cache.append(buffer, fin.gcount());
      }
      LoDTensor* tensor = scope.FindVar(name)->GetMutable<LoDTensor>();
      tensor->DeserializeFromString(cache, dev_ctx.GetPlace());
      fin.close();
    }
    VLOG(1) << "Complete loading variables.";
  }
};

class RestoreOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RestoreOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out",
              "(tensor), the tensor count can be 1~INT_MAX, tensors which "
              "values will be restores.")
        .AsDuplicable();
    AddAttr<std::string>("folderPath", "the folderPath for model file.");
    AddAttr<int>("data_type", "output tensor data type")
        .SetDefault(framework::DataType::FP32);
    AddComment(R"DOC(
Restore the tensors from model file based on absolute path.

All the tensors outputs may carry the LoD (Level of Details) information,
or not.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(save, paddle::operators::SaveOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::SaveOpMaker);

REGISTER_OPERATOR(restore, paddle::operators::RestoreOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::RestoreOpMaker);
