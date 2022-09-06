/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// disable numpy compile error

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <Python.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/slice_utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/detail/internals.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace pybind {

extern PyTypeObject* p_tensor_type;

bool PyCheckTensor(PyObject* obj) {
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(p_tensor_type));
}

static PyObject* tensor_add_method(PyObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  EAGER_TRY
  VLOG(6) << "Running tensor_add_method";

  paddle::experimental::Tensor ret =
      add_dygraph_function(CastPyArg2Tensor(PyTuple_GET_ITEM(args, 0), 0),
                           CastPyArg2Tensor(PyTuple_GET_ITEM(args, 1), 1));

  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef math_op_patch_function[] = {
    {"__add__",
     (PyCFunction)(void (*)(void))tensor_add_method,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {NULL, NULL, 0, NULL}};

void BindMathFunctions(PyObject* module) {
  if (PyModule_AddFunctions(module, math_op_patch_function) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindMathFunctions(PyModule_AddFunctions)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
