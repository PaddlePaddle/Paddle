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

#include <algorithm>
#include <ctime>

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

#define CLOG std::cout

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(char)};
  framework::LoD lod;
  int summarize;
  void* data{nullptr};

  void operator()(size_t size) {
    PrintMessage();
    PrintName();
    PrintDims();
    PrintDtype();
    PrintLod();
    PrintData(size);
  }

 private:
  void PrintMessage() { CLOG << std::time(nullptr) << "\t" << message; }
  void PrintName() {
    if (!name.empty()) {
      CLOG << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      CLOG << "\tshape: [";
      for (auto i : dims) {
        CLOG << i << ",";
      }
      CLOG << "]" << std::endl;
    }
  }
  void PrintDtype() {
    if (dtype.hash_code() != typeid(char).hash_code()) {
      CLOG << "\tdtype: " << dtype.name() << std::endl;
    }
  }
  void PrintLod() {
    if (!lod.empty()) {
      CLOG << "\tLoD: [";
      for (auto level : lod) {
        CLOG << "[ ";
        for (auto i : level) {
          CLOG << i << ",";
        }
        CLOG << " ]";
      }
      CLOG << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (dtype.hash_code() == typeid(float).hash_code()) {
      Display<float>(size);
    }
    if (dtype.hash_code() == typeid(double).hash_code()) {
      Display<double>(size);
    }
    if (dtype.hash_code() == typeid(int).hash_code()) {
      Display<int>(size);
    }
    if (dtype.hash_code() == typeid(int64_t).hash_code()) {
      Display<int64_t>(size);
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto* d = (T*)data;
    CLOG << "\tdata: ";
    if (summarize != -1) {
      summarize = std::min(size, (size_t)summarize);
      for (int i = 0; i < summarize; i++) {
        CLOG << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        CLOG << d[i] << ",";
      }
    }
    CLOG << std::endl;
  }
};

// TODO(ChunweiYan) there should be some other printers for TensorArray
class TensorPrintOp : public framework::OperatorBase {
 public:
  TensorPrintOp(const std::string& type,
                const framework::VariableNameMap& inputs,
                const framework::VariableNameMap& outputs,
                const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  TensorPrintOp(const TensorPrintOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::Place& place) const override {
    // Only run the `first_n` times.
    int first_n = Attr<int>("first_n");
    if (first_n > 0 && ++times_ > first_n) return;

    PADDLE_ENFORCE(!Inputs("input").empty(), "input should be set");
    auto* input_var = scope.FindVar(Input("input"));
    PADDLE_ENFORCE_NOT_NULL(input_var);
    auto& tensor = input_var->Get<framework::LoDTensor>();

    // TODO(ChunweiYan) support GPU
    PADDLE_ENFORCE(platform::is_cpu_place(tensor.place()));

    Formater formater;
    if (Attr<bool>("print_tensor_name")) {
      formater.name = Inputs("input").front();
    }
    if (Attr<bool>("print_tensor_type")) {
      formater.dtype = tensor.type();
    }
    if (Attr<bool>("print_tensor_shape")) {
      formater.dims.assign(tensor.dims()[0],
                           tensor.dims()[tensor.dims().size() - 1]);
    }
    if (Attr<bool>("print_tensor_lod")) {
      formater.lod = tensor.lod();
    }
    formater.summarize = Attr<int>("summarize");
    formater.data = (void*)tensor.data<void>();
    formater(tensor.numel());
  }

 private:
  mutable int times_{0};
};

class PrintOpProtoAndCheckMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PrintOpProtoAndCheckMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "the tensor that will be displayed.");
    AddAttr<int>("first_n", "Only log `first_n` number of times.");
    AddAttr<std::string>("message", "A string message to print as a prefix.");
    AddAttr<int>("summarize", "Print this number of elements in the tensor.");
    AddAttr<bool>("print_tensor_name", "Whether to print the tensor name.");
    AddAttr<bool>("print_tensor_type", "Whether to print the tensor's dtype.");
    AddAttr<bool>("print_tensor_shape", "Whether to print the tensor's shape.");
    AddAttr<bool>("print_tensor_lod", "Whether to print the tensor's lod.");
    AddComment(R"DOC(
    Creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.)DOC");
  }
};

class InferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("input"), "input should be set");
  }
};

class InferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {}
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(print, paddle::operators::TensorPrintOp,
                  paddle::operators::PrintOpProtoAndCheckMaker,
                  paddle::operators::InferShape,
                  paddle::operators::InferVarType,
                  paddle::framework::EmptyGradOpMaker);
