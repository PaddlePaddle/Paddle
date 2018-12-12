// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/py_func_op.h"
#include <set>
#include <string>
#include <vector>
#include "Python.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

namespace py = pybind11;

static std::vector<py::object> g_py_callables;

size_t AppendPythonCallableObjectAndReturnId(py::object py_obj) {
  g_py_callables.emplace_back(py_obj);
  return g_py_callables.size() - 1;
}

static py::object *GetPythonCallableObject(size_t i) {
  PADDLE_ENFORCE_LT(i, g_py_callables.size());
  return &g_py_callables[i];
}

void CallPythonFunc(py::object *callable, const std::string &func_token,
                    const std::vector<framework::LoDTensor> &ins,
                    std::vector<framework::LoDTensor *> *out) {
  py::gil_scoped_acquire guard{};
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = ins[i].IsInitialized() ? py::cast(ins[i]) : py::cast(nullptr);
  }

  auto ret = (*callable)(func_token, *in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  PADDLE_ENFORCE_EQ(py::len(ret_tuple), out->size(), "Output number not match");
  for (size_t i = 0; i < out->size(); ++i) {
    if ((*out)[i] == nullptr) {
      continue;
    }
    try {
      auto *out_tensor = py::cast<framework::LoDTensor *>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(out_tensor,
                              "Output tensor should not be nullptr");
      (*out)[i]->set_lod(out_tensor->lod());
      (*out)[i]->ShareDataWith(*out_tensor);
    } catch (py::cast_error &) {
      PADDLE_THROW("Output %d is not LoDTensor", i);
    }
  }
}

class PyFuncOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(!ctx->IsRuntime(),
                   "Infer shape cannot be called in runtime.");
    PADDLE_ENFORCE(ctx->HasInputs("X"), "Input(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"), "Output(Out) must exist");

    auto *op = boost::get<const framework::OpDesc *>(ctx->GetOp());
    auto *block = op->Block();
    // No need to infer shape in forward part
    if (block->ForwardBlockID() < 0) {
      return;
    }

    PADDLE_ENFORCE(!ctx->Attrs().Get<std::string>("token").empty(),
                   "Function token cannot be empty");

    const std::string kGradVarSuffix = framework::kGradVarSuffix;
    auto out_vars = ctx->GetOutputVarPtrs("Out");
    for (auto &out_var : out_vars) {
      auto *out_var_desc = boost::get<framework::VarDesc *>(out_var);
      auto out_name = out_var_desc->Name();
      if (out_name == framework::kEmptyVarName ||
          out_name.size() < kGradVarSuffix.size()) {
        continue;
      }

      size_t len = out_name.size() - kGradVarSuffix.size();
      if (out_name.substr(len) == kGradVarSuffix) {
        auto fwd_var_name = out_name.substr(0, len);
        auto *in_var_desc = block->FindVarRecursive(fwd_var_name);
        PADDLE_ENFORCE_NOT_NULL(in_var_desc, "Forward variable %s not found",
                                fwd_var_name);
        out_var_desc->SetShape(in_var_desc->GetShape());
        out_var_desc->SetDataType(in_var_desc->GetDataType());
        out_var_desc->SetLoDLevel(in_var_desc->GetLoDLevel());
        out_var_desc->SetType(in_var_desc->GetType());
      }
    }
  }
};

class PyFuncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Inputs of py_func op.").AsDuplicable();
    AddOutput("Out", "Outputs of py_func op").AsDuplicable();
    AddAttr<int>("handle_idx", "Index of the registered py_func handle")
        .SetDefault(0);
    AddAttr<std::string>("token", "Token of function token to be called")
        .SetDefault("");
    AddAttr<std::string>("backward_token",
                         "Token of backward function to be called")
        .SetDefault("");
    AddComment(R"DOC("PyFunc Op")DOC");
  }
};

class PyFuncOpGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    auto &fwd_attrs = Attrs();
    if (fwd_attrs.at("backward_token").empty()) {
      return {};
    }

    std::unique_ptr<framework::OpDesc> grad_op(new framework::OpDesc());
    grad_op->SetType("py_func");

    framework::AttributeMap bwd_attrs;
    bwd_attrs["token"] = fwd_attrs.at("backward_token");
    bwd_attrs["backward_token"] = std::string("");
    grad_op->SetAttrMap(bwd_attrs);

    auto bwd_in = Input("X");
    auto fwd_out = Output("Out");
    auto fwd_out_grad = OutputGrad("Out");
    bwd_in.insert(bwd_in.end(), fwd_out.begin(), fwd_out.end());
    bwd_in.insert(bwd_in.end(), fwd_out_grad.begin(), fwd_out_grad.end());

    auto bwd_out = InputGrad("X", false);

    if (VLOG_IS_ON(10)) {
      std::string in_str = "PyFunc Grad Input: ";
      for (auto &in : bwd_in) {
        in_str += in;
        in_str += " ";
      }
      VLOG(10) << in_str;

      std::string out_str = "PyFunc Grad Output: ";
      for (auto &out : bwd_out) {
        out_str += out;
        out += " ";
      }
      VLOG(10) << out_str;
    }

    grad_op->SetInput("X", bwd_in);
    grad_op->SetOutput("Out", InputGrad("X", false));

    std::vector<std::unique_ptr<framework::OpDesc>> ret(1);
    ret[0] = std::move(grad_op);
    return ret;
  }
};

class PyFuncOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &in_arg_names = Inputs("X");
    auto &out_arg_names = Outputs("Out");

    std::vector<framework::LoDTensor> inputs(in_arg_names.size());
    for (size_t i = 0; i < in_arg_names.size(); ++i) {
      auto in_var = scope.FindVar(in_arg_names[i]);
      if (in_var == nullptr) {
        continue;
      }
      auto &in_tensor = in_var->Get<framework::LoDTensor>();
      if (!in_tensor.IsInitialized()) {
        continue;
      }
      if (platform::is_gpu_place(in_tensor.place())) {
        framework::TensorCopySync(in_tensor, platform::CPUPlace(), &inputs[i]);
      } else {
        inputs[i].ShareDataWith(in_tensor);
      }
      inputs[i].set_lod(in_tensor.lod());
    }

    std::vector<framework::LoDTensor *> outputs(out_arg_names.size());
    for (size_t i = 0; i < out_arg_names.size(); ++i) {
      auto *out_var = scope.FindVar(out_arg_names[i]);
      auto *out_tensor =
          out_var ? out_var->GetMutable<framework::LoDTensor>() : nullptr;
      outputs[i] = out_tensor;
    }

    auto &token = Attr<std::string>("token");
    auto handle_idx = static_cast<size_t>(Attr<int>("handle_idx"));
    auto *py_callable = GetPythonCallableObject(handle_idx);
    VLOG(10) << "Call py_func_op with token " << token << ", and handle_idx "
             << handle_idx;
    CallPythonFunc(py_callable, token, inputs, &outputs);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(py_func, ops::PyFuncOp, ops::PyFuncOpMaker,
                  ops::PyFuncOpShapeInference, ops::PyFuncOpGradDescMaker);
