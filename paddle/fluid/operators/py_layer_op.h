// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/python_headers.h"

namespace paddle {
namespace operators {
namespace py = ::pybind11;

class PyLayerContext {
 public:
  explicit PyLayerContext(PyObject* context) : context_(context) {
    Py_INCREF(context_);
  }

  PyLayerContext() = delete;

  PyObject* GetMutableCtx() { return context_; }
  ~PyLayerContext() {
    py::gil_scoped_acquire guard;
    Py_XDECREF(context_);
  }

 private:
  PyObject* context_;
};

class PyLayerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(3) << "`InferShape` of `PyLayer` is an empty function, and it cannot "
               "infer the shape of the output tensors.";
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    return framework::OpKernelType(data_type, ctx.device_context());
  }

 public:
  void SetPyLayerContext(const std::shared_ptr<PyLayerContext>& py_context) {
    py_context_ = py_context;
  }
  std::shared_ptr<PyLayerContext> ReleasePyLayerContext() {
    auto temp = py_context_;
    py_context_.reset();
    VLOG(3) << "`py_context_` in the PyLayerOp is released.";
    return temp;
  }

 private:
  std::shared_ptr<PyLayerContext> py_context_;
};

template <typename T>
class PyLayerGradOpMaker {};
template <>
class PyLayerGradOpMaker<paddle::framework::OpDesc>
    : public framework::SingleGradOpMaker<paddle::framework::OpDesc> {
 public:
  using framework::SingleGradOpMaker<
      paddle::framework::OpDesc>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::framework::OpDesc> grad_op) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "`PyLayer` don't support static graph mode."));
  }
};

template <>
class PyLayerGradOpMaker<paddle::imperative::OpBase>
    : public framework::SingleGradOpMaker<paddle::imperative::OpBase> {
 public:
  using framework::SingleGradOpMaker<
      paddle::imperative::OpBase>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::imperative::OpBase> grad_op) const override;

 public:
  void SetPyLayerContext(const std::shared_ptr<PyLayerContext>& py_context) {
    py_context_ = py_context;
  }

 private:
  std::shared_ptr<PyLayerContext> py_context_;
};

}  // namespace operators
}  // namespace paddle
