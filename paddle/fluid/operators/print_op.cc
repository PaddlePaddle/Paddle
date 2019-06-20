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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {
using framework::GradVarName;

#define CLOG std::cout

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";
const char kBoth[] = "BOTH";

class LogGuard {
 public:
  inline LogGuard() { LogMutex().lock(); }

  inline ~LogGuard() { LogMutex().unlock(); }

 private:
  static std::mutex &LogMutex() {
    static std::mutex mtx;
    return mtx;
  }
};

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(const char)};
  framework::LoD lod;
  int summarize;
  void *data{nullptr};
  platform::Place place;
  std::stringstream logs;

  void operator()(size_t size) {
    PrintMessage();
    PrintPlaceInfo();
    PrintName();
    PrintDims();
    PrintDtype();
    PrintLod();
    PrintData(size);
    LogGuard guard;
    CLOG << logs.str();
  }

 private:
  void PrintPlaceInfo() { logs << "The place is:" << place << std::endl; }
  void PrintMessage() { logs << std::time(nullptr) << "\t" << message << "\t"; }
  void PrintName() {
    if (!name.empty()) {
      logs << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      logs << "\tshape: [";
      for (auto i : dims) {
        logs << i << ",";
      }
      logs << "]" << std::endl;
    }
  }
  void PrintDtype() {
    if (!framework::IsType<const char>(dtype)) {
      logs << "\tdtype: " << dtype.name() << std::endl;
    }
  }
  void PrintLod() {
    if (!lod.empty()) {
      logs << "\tLoD: [";
      for (auto level : lod) {
        logs << "[ ";
        for (auto i : level) {
          logs << i << ",";
        }
        logs << " ]";
      }
      logs << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (framework::IsType<const float>(dtype)) {
      Display<float>(size);
    } else if (framework::IsType<const double>(dtype)) {
      Display<double>(size);
    } else if (framework::IsType<const int>(dtype)) {
      Display<int>(size);
    } else if (framework::IsType<const int64_t>(dtype)) {
      Display<int64_t>(size);
    } else if (framework::IsType<const bool>(dtype)) {
      Display<bool>(size);
    } else {
      logs << "\tdata: unprintable type: " << dtype.name() << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto *d = reinterpret_cast<T *>(data);
    logs << "\tdata: ";
    if (summarize != -1) {
      summarize = std::min(size, (size_t)summarize);
      for (int i = 0; i < summarize; i++) {
        logs << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        logs << d[i] << ",";
      }
    }
    logs << std::endl;
  }
};

// TODO(ChunweiYan) there should be some other printers for TensorArray
class PrintOp : public framework::OperatorBase {
 public:
  PrintOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    const auto in_var = scope.FindVar(Input("In"));
    auto out_var = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE_NOT_NULL(in_var, "The input should not be found in scope",
                            Input("In"));
    PADDLE_ENFORCE_NOT_NULL(out_var, "The output should not be found in scope",
                            Output("Out"));
    auto &in_tensor = in_var->Get<framework::LoDTensor>();
    framework::LoDTensor *out_tensor =
        out_var->GetMutable<framework::LoDTensor>();

    PrintValue(place, Inputs("In").front(), in_tensor);
    framework::TensorCopy(in_tensor, place, out_tensor);
    out_tensor->set_lod(in_tensor.lod());
  }

  void PrintValue(const platform::Place &place,
                  const std::string &printed_var_name,
                  const framework::LoDTensor &in_tensor) const {
    std::string print_phase = Attr<std::string>("print_phase");
    bool is_forward = Attr<bool>("is_forward");

    if ((is_forward && print_phase == kBackward) ||
        (!is_forward && print_phase == kForward)) {
      return;
    }

    int first_n = Attr<int>("first_n");
    if (first_n > 0 && ++times_ > first_n) return;

    framework::LoDTensor printed_tensor;
    printed_tensor.set_lod(in_tensor.lod());
    printed_tensor.Resize(in_tensor.dims());

    if (is_cpu_place(in_tensor.place())) {
      printed_tensor.ShareDataWith(in_tensor);
    } else {
      // copy data to cpu to print
      platform::CPUPlace place;
      TensorCopy(in_tensor, place, &printed_tensor);
    }

    Formater formater;
    formater.place = place;
    formater.message = Attr<std::string>("message");
    if (Attr<bool>("print_tensor_name")) {
      formater.name = printed_var_name;
    }
    if (Attr<bool>("print_tensor_type")) {
      formater.dtype = framework::ToTypeIndex(printed_tensor.type());
    }
    if (Attr<bool>("print_tensor_shape")) {
      auto &dims = printed_tensor.dims();
      formater.dims.resize(dims.size());
      for (int i = 0; i < dims.size(); ++i) formater.dims[i] = dims[i];
    }
    if (Attr<bool>("print_tensor_lod")) {
      formater.lod = printed_tensor.lod();
    }
    formater.summarize = Attr<int>("summarize");
    formater.data = reinterpret_cast<void *>(printed_tensor.data<void>());
    formater(printed_tensor.numel());
  }

 private:
  mutable int times_{0};
};

class PrintOpProtoAndCheckMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("In", "Input tensor to be displayed.");
    AddOutput("Out", "The output tensor.");
    AddAttr<int>("first_n", "Only log `first_n` number of times.");
    AddAttr<std::string>("message", "A string message to print as a prefix.");
    AddAttr<int>("summarize", "Number of elements printed.");
    AddAttr<bool>("print_tensor_name", "Whether to print the tensor name.");
    AddAttr<bool>("print_tensor_type", "Whether to print the tensor's dtype.");
    AddAttr<bool>("print_tensor_shape", "Whether to print the tensor's shape.");
    AddAttr<bool>("print_tensor_lod", "Whether to print the tensor's lod.");
    AddAttr<std::string>("print_phase",
                         "(string, default 'FORWARD') Which phase to display "
                         "including 'FORWARD' "
                         "'BACKWARD' and 'BOTH'.")
        .SetDefault(std::string(kBoth))
        .InEnum({std::string(kForward), std::string(kBackward),
                 std::string(kBoth)});
    AddAttr<bool>("is_forward", "Whether is forward or not").SetDefault(true);
    AddComment(R"DOC(
Creates a print op that will print when a tensor is accessed.

Wraps the tensor passed in so that whenever that a tensor is accessed,
the message `message` is printed, along with the current value of the
tensor `t`.)DOC");
  }
};

class PrintOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    VLOG(10) << "PrintOpInferShape";
    PADDLE_ENFORCE(ctx->HasInput("In"), "Input(In) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null.");
    ctx->ShareDim("In", /*->*/ "Out");
    ctx->ShareLoD("In", /*->*/ "Out");
  }
};

class PrintOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto input_type = ctx->GetType(ctx->Input("In")[0]);
    auto out_name = ctx->Output("Out").front();
    ctx->SetType(out_name, input_type);
  }
};

class PrintOpGradientMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType("print");
    op_desc_ptr->SetInput("In", OutputGrad("Out"));
    op_desc_ptr->SetOutput("Out", InputGrad("In"));
    op_desc_ptr->SetAttrMap(Attrs());
    op_desc_ptr->SetAttr("is_forward", false);
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(print, ops::PrintOp, ops::PrintOpProtoAndCheckMaker,
                  ops::PrintOpGradientMaker, ops::PrintOpInferShape,
                  ops::PrintOpVarTypeInference);
