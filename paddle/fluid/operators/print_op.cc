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

#include <algorithm>
#include <ctime>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {

#define CLOG std::cout

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";
const char kBoth[] = "BOTH";

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(const char)};
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
  void PrintMessage() { CLOG << std::time(nullptr) << "\t" << message << "\t"; }
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
    if (dtype.hash_code() != typeid(const char).hash_code()) {
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
    if (dtype.hash_code() == typeid(const float).hash_code()) {
      Display<float>(size);
    } else if (dtype.hash_code() == typeid(const double).hash_code()) {
      Display<double>(size);
    } else if (dtype.hash_code() == typeid(const int).hash_code()) {
      Display<int>(size);
    } else if (dtype.hash_code() == typeid(const int64_t).hash_code()) {
      Display<int64_t>(size);
    } else if (dtype.hash_code() == typeid(const bool).hash_code()) {
      Display<bool>(size);
    } else {
      CLOG << "\tdata: unprintable type: " << dtype.name() << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto* d = reinterpret_cast<T*>(data);
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
    PADDLE_THROW("Not implemented.");
  }

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    const framework::Variable* in_var_ptr = nullptr;
    std::string phase(kForward);
    std::string printed_var_name = "";

    auto& inputs = Inputs();
    if (inputs.find("In") != inputs.end() && !Inputs("In").empty()) {
      in_var_ptr = scope.FindVar(Input("In"));
      printed_var_name = Inputs("In").front();
    } else if (inputs.find("In@GRAD") != inputs.end() &&
               !Inputs("In@GRAD").empty()) {
      in_var_ptr = scope.FindVar(Input("In@GRAD"));
      printed_var_name = Inputs("In@GRAD").front();
      phase = std::string(kBackward);
    } else {
      PADDLE_THROW("Unknown phase, should be forward or backward.");
    }

    PADDLE_ENFORCE_NOT_NULL(in_var_ptr);

    auto& in_tensor = in_var_ptr->Get<framework::LoDTensor>();
    auto* out_var_ptr = scope.FindVar(Output("Out"));
    auto& out_tensor = *out_var_ptr->GetMutable<framework::LoDTensor>();

    // Just copy data from input tensor to output tensor
    // output tensor share same memory with input tensor
    out_tensor.ShareDataWith(in_tensor);
    out_tensor.set_lod(in_tensor.lod());

    std::string print_phase = Attr<std::string>("print_phase");
    if (print_phase != phase && print_phase != std::string(kBoth)) {
      return;
    }

    int first_n = Attr<int>("first_n");
    if (first_n > 0 && ++times_ > first_n) return;

    framework::LoDTensor printed_tensor;
    printed_tensor.set_lod(in_tensor.lod());
    printed_tensor.Resize(in_tensor.dims());

    if (platform::is_cpu_place(in_tensor.place())) {
      printed_tensor.ShareDataWith(in_tensor);
    } else {
      // copy data to cpu to print
      platform::CPUPlace place;
      framework::TensorCopy(in_tensor, place, &printed_tensor);
    }

    Formater formater;
    formater.message = Attr<std::string>("message");
    if (Attr<bool>("print_tensor_name")) {
      formater.name = printed_var_name;
    }
    if (Attr<bool>("print_tensor_type")) {
      formater.dtype = printed_tensor.type();
    }
    if (Attr<bool>("print_tensor_shape")) {
      auto& dims = printed_tensor.dims();
      formater.dims.resize(dims.size());
      for (int i = 0; i < dims.size(); ++i) formater.dims[i] = dims[i];
    }
    if (Attr<bool>("print_tensor_lod")) {
      formater.lod = printed_tensor.lod();
    }
    formater.summarize = Attr<int>("summarize");
    formater.data = reinterpret_cast<void*>(printed_tensor.data<void>());
    formater(printed_tensor.numel());
  }

 private:
  mutable int times_{0};
};

class PrintOpProtoAndCheckMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("In", "Input tensor to be displayed.");
    AddAttr<int>("first_n", "Only log `first_n` number of times.");
    AddAttr<std::string>("message", "A string message to print as a prefix.");
    AddAttr<int>("summarize", "Number of elements printed.");
    AddAttr<bool>("print_tensor_name", "Whether to print the tensor name.");
    AddAttr<bool>("print_tensor_type", "Whether to print the tensor's dtype.");
    AddAttr<bool>("print_tensor_shape", "Whether to print the tensor's shape.");
    AddAttr<bool>("print_tensor_lod", "Whether to print the tensor's lod.");
    AddAttr<std::string>(
        "print_phase",
        "(string, default 'BOTH') Which phase to display including 'FORWARD' "
        "'BACKWARD' and 'BOTH'.")
        .SetDefault(std::string(kBoth))
        .InEnum({std::string(kForward), std::string(kBackward),
                 std::string(kBoth)});
    AddOutput("Out", "Output tensor with same data as input tensor.");
    AddComment(R"DOC(
Creates a print op that will print when a tensor is accessed.

Wraps the tensor passed in so that whenever that a tensor is accessed,
the message `message` is printed, along with the current value of the
tensor `t`.)DOC");
  }
};

class InferShapeForward : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("In"), "Input(In) should not be null.");
    context->ShareLoD("In", /*->*/ "Out");
    context->SetOutputDim("Out", context->GetInputDim("In"));
  }
};

class InferShapeBackward : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("In@GRAD"),
                   "Input(In@GRAD) should not be null.");
    context->ShareLoD("In@GRAD", /*->*/ "Out");
    context->SetOutputDim("Out", context->GetInputDim("In@GRAD"));
  }
};

class InferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {}
};

class PrintOpProtoAndCheckGradOpMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType("print_grad");
    op_desc_ptr->SetInput("In@GRAD", OutputGrad("Out"));
    op_desc_ptr->SetOutput("Out", InputGrad("In"));
    op_desc_ptr->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(print, ops::TensorPrintOp, ops::PrintOpProtoAndCheckMaker,
                  ops::PrintOpProtoAndCheckGradOpMaker, ops::InferShapeForward,
                  ops::InferVarType);
REGISTER_OPERATOR(print_grad, ops::TensorPrintOp, ops::InferShapeBackward);
