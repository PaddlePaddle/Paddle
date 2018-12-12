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

namespace py = ::pybind11;

static std::vector<py::object> g_py_callables;

const char kForwardPythonCallableId[] = "forward_callable_id";
const char kBackwardPythonCallableId[] = "backward_callable_id";
const char kPyFuncBackwardSkipVars[] = "backward_skip_vars";

size_t AppendPythonCallableObjectAndReturnId(const py::object &py_obj) {
  g_py_callables.emplace_back(py_obj);
  return g_py_callables.size() - 1;
}

static py::object *GetPythonCallableObject(size_t i) {
  PADDLE_ENFORCE_LT(i, g_py_callables.size(), "Invalid python callable id");
  return &g_py_callables[i];
}

std::string PythonObjectToString(const py::object &py_callable) {
  py::gil_scoped_acquire guard;
  return py::str(*py_callable);
}

void CallPythonFunc(py::object *callable,
                    const std::vector<framework::LoDTensor> &ins,
                    std::vector<framework::LoDTensor *> *out) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = ins[i].IsInitialized() ? py::cast(ins[i]) : py::cast(nullptr);
  }

  auto ret = (*callable)(*in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  PADDLE_ENFORCE_EQ(py::len(ret_tuple), out->size(), "Output number not match");
  for (size_t i = 0; i < out->size(); ++i) {
    if ((*out)[i] == nullptr) {
      continue;
    }
    try {
      auto *out_tensor = py::cast<framework::LoDTensor *>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(out_tensor,
                              "Output tensor %d should not be nullptr", i);
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
    PADDLE_ENFORCE(ctx->HasInputs("X") || ctx->HasOutputs("Out"),
                   "Input(X) or Output(Out) must exist");
    PADDLE_ENFORCE_GE(ctx->Attrs().Get<int>(kForwardPythonCallableId), 0,
                      "Function id cannot be less than 0");

    auto *op = boost::get<const framework::OpDesc *>(ctx->GetOp());
    auto *block = op->Block();
    const std::string kGradVarSuffix = framework::kGradVarSuffix;
    auto out_vars = ctx->GetOutputVarPtrs("Out");
    for (auto &out_var : out_vars) {
      auto *out_var_desc = boost::get<framework::VarDesc *>(out_var);
      if (out_var_desc == nullptr) {
        continue;
      }
      auto out_name = out_var_desc->Name();
      if (out_name == framework::kEmptyVarName ||
          out_name.size() <= kGradVarSuffix.size()) {
        continue;
      }

      size_t len = out_name.size() - kGradVarSuffix.size();
      if (out_name.substr(len) == kGradVarSuffix) {
        auto fwd_var_name = out_name.substr(0, len);
        auto *in_var_desc = block->FindVarRecursive(fwd_var_name);
        PADDLE_ENFORCE_NOT_NULL(in_var_desc, "Forward variable %s not found",
                                fwd_var_name);
        VLOG(10) << "Infer shape of Out(" << out_name << ") as Input("
                 << in_var_desc->Name() << ")";
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
    AddAttr<int>(kForwardPythonCallableId,
                 "Index of registered forward Python function.")
        .SetDefault(0);
    AddAttr<int>(kBackwardPythonCallableId,
                 "Index of registered backward Python function")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>(kPyFuncBackwardSkipVars,
                                      "Unused forward in/out in backward op")
        .SetDefault(std::vector<std::string>());
    AddComment(R"DOC("PyFunc Op")DOC");
  }
};

class PyFuncOpGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    auto &fwd_attrs = Attrs();
    // no backward op when backward_id is less than 0
    if (boost::get<int>(fwd_attrs.at(kBackwardPythonCallableId)) < 0) {
      return {};
    }

    std::unique_ptr<framework::OpDesc> grad_op(new framework::OpDesc());
    grad_op->SetType("py_func");

    framework::AttributeMap bwd_attrs;
    bwd_attrs[kForwardPythonCallableId] =
        fwd_attrs.at(kBackwardPythonCallableId);
    bwd_attrs[kBackwardPythonCallableId] = -1;
    grad_op->SetAttrMap(bwd_attrs);

    // All forward inputs
    auto fwd_ins = Input("X");
    // All forward outputs
    auto fwd_outs = Output("Out");

    // For memory reused, some inputs/output in forward part may be not needed
    // in backward part
    // Just skip these vars
    auto &backward_skip_var_list = boost::get<std::vector<std::string>>(
        fwd_attrs.at(kPyFuncBackwardSkipVars));
    std::unordered_set<std::string> backward_skip_var_set(
        backward_skip_var_list.begin(), backward_skip_var_list.end());
    std::vector<std::string> bwd_ins;
    bwd_ins.reserve(fwd_ins.size() + fwd_outs.size());
    for (auto &fwd_in : fwd_ins) {
      if (backward_skip_var_set.count(fwd_in) == 0) {
        bwd_ins.emplace_back(fwd_in);
      }
    }

    for (auto &fwd_out : fwd_outs) {
      if (backward_skip_var_set.count(fwd_out) == 0) {
        bwd_ins.emplace_back(fwd_out);
      }
    }

    // Backward OG cannot be skipped
    // But in Python side, if OG is kEmptyVarName, input tensor would be None
    auto fwd_out_grads = OutputGrad("Out");
    bwd_ins.reserve(bwd_ins.size() + fwd_out_grads.size());
    bwd_ins.insert(bwd_ins.end(), fwd_out_grads.begin(), fwd_out_grads.end());

    // Backward IG cannot be skipped
    // But in Python side, if IG is not needed, users can just return None
    auto bwd_outs = InputGrad("X", false);

    if (VLOG_IS_ON(10)) {
      std::string in_str = "PyFunc Grad Input: ";
      for (auto &in : bwd_ins) {
        in_str += in;
        in_str += " ";
      }
      VLOG(10) << in_str;

      std::string out_str = "PyFunc Grad Output: ";
      for (auto &out : bwd_outs) {
        out_str += out;
        out_str += " ";
      }
      VLOG(10) << out_str;
    }

    grad_op->SetInput("X", bwd_ins);
    grad_op->SetOutput("Out", bwd_outs);

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

    auto callable_id = static_cast<size_t>(Attr<int>(kForwardPythonCallableId));
    auto *py_callable = GetPythonCallableObject(callable_id);
    VLOG(10) << "Call py_func_op with id " << callable_id << ": "
             << PythonObjectToString(*py_callable);
    CallPythonFunc(py_callable, inputs, &outputs);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(py_func, ops::PyFuncOp, ops::PyFuncOpMaker,
                  ops::PyFuncOpShapeInference, ops::PyFuncOpGradDescMaker);
