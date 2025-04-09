// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/pybind/python_callable_registry.h"
#include "glog/logging.h"  // For VLOG()

#include "paddle/common/errors.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

void PirCallPythonFunc(py::object *callable,
                       const std::vector<pir::Value> &ins,
                       std::vector<pir::Value> *outs) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = py::cast(ins[i]);
  }

  VLOG(2) << "[PirCallPythonFunc] ins.size() = " << ins.size();
  VLOG(2) << "[PirCallPythonFunc] outs->size() = " << outs->size();

  auto ret = (*callable)(*in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  size_t out_num = outs->size();
  if (UNLIKELY(ret_num != out_num)) {
    // Python function has no return values or returns None
    // In this case, ret_num = 1 && ret[0] == None && out_num should be 0
    // Otherwise, ret_num must be equal to out_num
    PADDLE_ENFORCE_EQ(ret_num == 1,
                      true,
                      common::errors::InvalidArgument(
                          "Python function has no return values or returns "
                          "None. In this case, ret_num = 1 && ret[0] == None "
                          "&& out_num should be 0. But ret_num is %d",
                          ret_num));

    PADDLE_ENFORCE_EQ(
        out_num == 0,
        true,
        common::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But out_num is %d",
            out_num));

    PADDLE_ENFORCE_EQ(
        py::cast<pir::Value *>(ret_tuple[0]) == nullptr,
        true,
        common::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But ret[0] is not None"));
  }

  for (size_t i = 0; i < out_num; ++i) {
    try {
      if (ret_tuple[i].is_none()) {
        VLOG(6) << "Set Output( " << i << " ) value as fake_value";
        (*outs)[i] = pir::Value(nullptr);
      } else {
        auto py_out_value = py::cast<pir::Value>(ret_tuple[i]);
        PADDLE_ENFORCE_NOT_NULL(
            py_out_value.impl(),
            common::errors::InvalidArgument(
                "Output value %d should not be nullptr", i));
        (*outs)[i] = py_out_value;
      }
    } catch (py::cast_error &) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "pybind11::cast to pir::Value error. The %d-th output exception is "
          "pir::Value",
          i));
    }
  }
}

PythonCallableRegistrar &PythonCallableRegistrar::GetInstance() {
  static thread_local PythonCallableRegistrar instance;
  return instance;
}
void PythonCallableRegistrar::Register(uint64_t unique_id,
                                       const py::object &callable) {
  if (python_callable_registry_.find(unique_id) !=
      python_callable_registry_.end()) {
    LOG(WARNING) << "unique_id " << unique_id
                 << " has already registered python function in "
                    "python_callable_registry_. This operation "
                    "will override the old one.";
  }
  python_callable_registry_[unique_id] = callable;
}

// Return pybind11::object* instead of pybind11::object
// Returning pybind11::object would cause reference count increasing
// but without GIL, reference count in Python may not be safe
py::object *PythonCallableRegistrar::Get(uint64_t unique_id) {
  PADDLE_ENFORCE_NE(
      python_callable_registry_.find(unique_id),
      python_callable_registry_.end(),
      common::errors::InvalidArgument(
          "Unique_id %d is not found in python_callable_registry_. The "
          "possible "
          "reasons are below:\n"
          "1. The python callable was not registered for `unique_id` by "
          "`PythonCallableRegistrar::Register`\n"
          "2. The python callable was remove from python_callable_registry_",
          unique_id));
  return &(python_callable_registry_[unique_id]);
}

}  // namespace pybind
}  // namespace paddle
