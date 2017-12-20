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
  void* data{nullptr};

  void operator()() {
    PrintMessage();
    PrintName();
    PrintDims();
    PrintDtype();
    PrintLod();
    PrintData();
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

  void PrintData() {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (dtype.hash_code() == typeid(float).hash_code()) {
      Display<float>();
    }
    if (dtype.hash_code() == typeid(double).hash_code()) {
      Display<double>();
    }
    if (dtype.hash_code() == typeid(int).hash_code()) {
      Display<int>();
    }
    if (dtype.hash_code() == typeid(int64_t).hash_code()) {
      Display<int64_t>();
    }
  }

  template <typename T>
  void Display() {
    auto* d = (T*)data;
    (void)d;
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
           const platform::DeviceContext& dev_ctx) const override {
    PADDLE_ENFORCE(!Inputs("input").empty(), "input should be set");
    auto* input_var = scope.FindVar(Input("input"));
    PADDLE_ENFORCE_NOT_NULL(input_var);
    auto& tensor = input_var->Get<framework::LoDTensor>();

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
    formater.data = (void*)tensor.data<void>();

    formater();
  }
};

class PrintOpProtoAndCheckMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PrintOpProtoAndCheckMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "the tensor that will be displayed.");
    AddOutput("output", "just a place holder");
    AddAttr<int>("first_n", "Only log `first_n` number of times.");
    AddAttr<std::string>("message", "A string message to print as a prefix.");
    AddAttr<bool>("print_tensor_name", "Whether to print the tensor name.");
    AddAttr<bool>("print_tensor_type", "Whether to print the tensor's dtype.");
    AddAttr<bool>("print_tensor_shape", "Whether to print the tensor's shape.");
    AddAttr<bool>("print_tensor_lod", "Whether to print the tensor's lod.");
    AddComment(R"DOC(
    Creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
      input: A Tensor to print.
      message: A string message to print as a prefix.
      first_n: Only log `first_n` number of times.
      print_tensor_name: Print the tensor name.
      print_tensor_type: Print the tensor type.
      print_tensor_shape: Print the tensor shape.
      print_tensor_lod: Print the tensor lod.
)DOC");
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
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {}
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(print, paddle::operators::TensorPrintOp,
                  paddle::operators::PrintOpProtoAndCheckMaker,
                  paddle::operators::InferShape,
                  paddle::operators::InferVarType,
                  paddle::framework::EmptyGradOpMaker);
