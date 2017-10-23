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

template <typename T>
class SaveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::string folder_path = ctx.template Attr<std::string>("folderPath");
    auto tensors = ctx.MultiInput<LoDTensor>("X");
    auto& var_names = ctx.Inputs("X");

    VLOG(1) << "Save variables to folder: " << folder_path;
    for (size_t i = 0; i < tensors.size(); ++i) {
      std::string file_name = VarToFileName(folder_path, var_names.at(i));
      std::ofstream fout(file_name, std::ofstream::out);
      PADDLE_ENFORCE(fout.is_open(), "Fail to create file %s.", file_name);
      std::string bytes = tensors[i]->SerializeToString();
      fout << bytes;
      fout.close();
    }
    VLOG(1) << "Compelete saving variables. Items count: " << tensors.size();
  }
};

template <typename Place, typename T>
class RestoreKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::string folder_path = ctx.template Attr<std::string>("folderPath");
    VLOG(1) << "Try loading variables from folder: " << folder_path;
    auto tensors = ctx.MultiOutput<LoDTensor>("Out");
    auto& var_names = ctx.Outputs("Out");

    for (size_t i = 0; i < var_names.size(); ++i) {
      std::string file_name = VarToFileName(folder_path, var_names[i]);
      std::ifstream fin(file_name, std::ifstream::in);
      PADDLE_ENFORCE(fin.is_open(), "Fail to open file %s.", file_name);
      const size_t kBufferSize = 4096;  // equal to linux page size
      char buffer[kBufferSize];
      std::string cache;
      while (!fin.eof()) {
        fin.read(buffer, kBufferSize);
        cache.append(buffer, fin.gcount());
      }
      tensors.at(i)->DeserializeFromString(cache, ctx.GetPlace());
      fin.close();
    }
    VLOG(1) << "Complete loading variables.";
  }
};

class SaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"),
                   "Input(X) of SaveOp should not be null.");
    auto folder_path = ctx->Attrs().Get<std::string>("folderPath");
    PADDLE_ENFORCE(!folder_path.empty(),
                   "Input(folder_path) of SaveOp should not be null.");
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

class RestoreOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutputs("Out"),
                   "Output(X) of RestoreOp should not be null.");
    auto folder_path = ctx->Attrs().Get<std::string>("folderPath");
    PADDLE_ENFORCE(!folder_path.empty(),
                   "Input(folder_path) of Restore Op should not be null.");
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return static_cast<framework::DataType>(Attr<int>("data_type"));
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

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(save, ops::SaveOp, ops::SaveOpMaker);
REGISTER_OP_CPU_KERNEL(save, ops::SaveKernel<float>);

REGISTER_OP_WITHOUT_GRADIENT(restore, ops::RestoreOp, ops::RestoreOpMaker);
REGISTER_OP_CPU_KERNEL(restore,
                       ops::RestoreKernel<paddle::platform::CPUPlace, float>);
