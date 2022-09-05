// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/saved_tensors_hooks.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"

namespace egr {

class PackHook : public PackHookBase {
 public:
  explicit PackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

  ~PackHook() {
    ::pybind11::gil_scoped_acquire gil;
    Py_DECREF(hook_);
  }

  void* operator()(const paddle::experimental::Tensor& tensor) override {
    auto args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, paddle::pybind::ToPyObject(tensor));
    bool grad_tmp = egr::Controller::Instance().HasGrad();
    egr::Controller::Instance().SetHasGrad(false);
    PyObject* ret = nullptr;
    {
      ::pybind11::gil_scoped_acquire gil;
      ret = PyObject_Call(hook_, args, nullptr);
    }
    Py_XDECREF(args);
    egr::Controller::Instance().SetHasGrad(grad_tmp);
    return reinterpret_cast<void*>(ret);
  }

  void* operator()(void* py_tensor) override {
    auto args = PyTuple_New(1);
    Py_INCREF(reinterpret_cast<PyObject*>(py_tensor));
    PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(py_tensor));
    bool grad_tmp = egr::Controller::Instance().HasGrad();
    egr::Controller::Instance().SetHasGrad(false);
    PyObject* ret = nullptr;
    {
      ::pybind11::gil_scoped_acquire gil;
      ret = PyObject_Call(hook_, args, nullptr);
    }
    Py_XDECREF(args);
    egr::Controller::Instance().SetHasGrad(grad_tmp);
    return reinterpret_cast<void*>(ret);
  }

 private:
  PyObject* hook_;
};

class UnPackHook : public UnPackHookBase {
 public:
  explicit UnPackHook(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

  ~UnPackHook() {
    ::pybind11::gil_scoped_acquire gil;
    Py_DECREF(hook_);
  }

  paddle::experimental::Tensor operator()(void* packed_value) override {
    auto args = PyTuple_New(1);
    Py_INCREF(reinterpret_cast<PyObject*>(packed_value));
    PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(packed_value));
    bool grad_tmp = egr::Controller::Instance().HasGrad();
    egr::Controller::Instance().SetHasGrad(false);
    PyObject* ret = nullptr;
    {
      ::pybind11::gil_scoped_acquire gil;
      ret = PyObject_Call(hook_, args, nullptr);
    }
    Py_XDECREF(args);
    egr::Controller::Instance().SetHasGrad(grad_tmp);

    PADDLE_ENFORCE_EQ(paddle::pybind::IsEagerTensor(ret),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "paddle.autograd.SavedTensorsHooks only one pair "
                          "of hooks is allowed at a time."));

    return reinterpret_cast<paddle::pybind::TensorObject*>(ret)->tensor;
  }

 private:
  PyObject* hook_;
};

void SavedTensorsHooks::SetHooks(PyObject* pack_hook, PyObject* unpack_hook) {
  PADDLE_ENFORCE_EQ(pack_hook_ == nullptr && unpack_hook_ == nullptr,
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.SavedTensorsHooks only one pair "
                        "of hooks is allowed at a time."));
  pack_hook_ = std::make_shared<PackHook>(pack_hook);
  unpack_hook_ = std::make_shared<UnPackHook>(unpack_hook);
  is_enable_ = true;
}

}  // namespace egr
