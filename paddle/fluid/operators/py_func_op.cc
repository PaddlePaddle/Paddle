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

static std::mutex g_py_callables_mtx;
static std::vector<py::object> g_py_callables;

size_t AppendPythonCallableObjectAndReturnId(py::object py_obj) {
  std::lock_guard<std::mutex> guard(g_py_callables_mtx);
  g_py_callables.emplace_back(py_obj);
  return g_py_callables.size() - 1;
}

static py::object *GetPythonCallableObject(size_t i) {
  std::lock_guard<std::mutex> guard(g_py_callables_mtx);
  PADDLE_ENFORCE_LT(i, g_py_callables.size());
  return &g_py_callables[i];
}

void DoCallPythonFunc(py::object *callable, const std::string &func_token,
                      const std::vector<framework::LoDTensor> &ins,
                      std::vector<framework::LoDTensor *> *out) {
  py::gil_scoped_acquire guard{};
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = py::cast(ins[i]);
  }

  auto ret = (*callable)(func_token, *in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  PADDLE_ENFORCE_EQ(py::len(ret_tuple), out->size(), "Output number not match");
  for (size_t i = 0; i < out->size(); ++i) {
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
    PADDLE_ENFORCE(ctx->HasInputs("X"), "Input(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"), "Output(Out) must exist");
  }
};

class PyFuncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Inputs of py_func op.").AsDuplicable();
    AddOutput("Out", "Outputs of py_func op").AsDuplicable();
    AddAttr<std::string>("token", "function token");
    AddAttr<int>("handle_idx", "handle index").SetDefault(0);
    AddComment(R"DOC("PyFunc Op")DOC");
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
      auto &in_tensor =
          scope.FindVar(in_arg_names[i])->Get<framework::LoDTensor>();
      if (platform::is_gpu_place(in_tensor.place())) {
        framework::TensorCopySync(in_tensor, platform::CPUPlace(), &inputs[i]);
      } else {
        inputs[i].ShareDataWith(in_tensor);
      }
      inputs[i].set_lod(in_tensor.lod());
    }

    std::vector<framework::LoDTensor *> outputs(out_arg_names.size());
    for (size_t i = 0; i < out_arg_names.size(); ++i) {
      auto *out_tensor =
          scope.FindVar(out_arg_names[i])->GetMutable<framework::LoDTensor>();
      outputs[i] = out_tensor;
    }

    auto &token = Attr<std::string>("token");
    auto handle_idx = static_cast<size_t>(Attr<int>("handle_idx"));
    auto *py_callable = GetPythonCallableObject(handle_idx);
    VLOG(10) << "Call py_func_op with token " << token << ", and handle_idx "
             << handle_idx;
    DoCallPythonFunc(py_callable, token, inputs, &outputs);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(py_func, ops::PyFuncOp, ops::PyFuncOpMaker,
                  ops::PyFuncOpShapeInference,
                  paddle::framework::EmptyGradOpMaker);
