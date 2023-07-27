// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/fluid/ir/dialect/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace pybind {

PyObject *static_api_mean(PyObject *self, PyObject *args, PyObject *kwargs) {
  try {
    VLOG(6) << "Add mean op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
    // Get OpResult from args
    PyObject *x_obj = PyTuple_GET_ITEM(args, 0);
    auto x = CastPyArg2OpResult("mean", x_obj, 0);

    // Parse Attributes if needed
    PyObject *axis_obj = PyTuple_GET_ITEM(args, 1);
    paddle::experimental::IntArray axis =
        CastPyArg2IntArray(axis_obj, "mean", 1);
    PyObject *keepdim_obj = PyTuple_GET_ITEM(args, 2);
    bool keepdim = CastPyArg2Boolean(keepdim_obj, "mean", 2);

    // Call ir static api
    auto out = paddle::dialect::mean(x, axis.GetData(), keepdim);
    return ToPyObject(out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

}  // namespace pybind
}  // namespace paddle
