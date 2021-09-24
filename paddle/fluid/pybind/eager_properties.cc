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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL Paddle_PyArray_API_P
#define INIT_NUMPY_ARRAY_CPP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/tcmpt/api/include/tensor.h"
#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/dtype.h"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wconversion-null"

namespace paddle {
namespace pybind {

int init_numpy_p() {
  import_array();
  return 0;
}
static const int numpy_initialized_m = init_numpy_p();

extern PyTypeObject* pEagerTensorType;

PyObject* eager_tensor_properties_get_shape(EagerTensorObject* self,
                                            void* closure) {
  auto ddim = self->eagertensor.shape();
  std::vector<int64_t> value;
  size_t rank = static_cast<size_t>(ddim.size());
  value.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    value[i] = ddim[i];
  }

  return ToPyObject(value);
}

PyObject* eager_tensor_properties_get_stop_gradient(EagerTensorObject* self,
                                                    void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  return ToPyObject(meta->StopGradient());
}

int eager_tensor_properties_set_stop_gradient(EagerTensorObject* self,
                                              PyObject* value, void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  meta->SetStopGradient(CastPyArg2AttrBoolean(value, 0));
  return 0;
}

PyObject* eager_tensor_properties_get_dtype(EagerTensorObject* self,
                                            void* closure) {
  return ToPyObject(DataType2String(self->eagertensor.type()));
}

PyObject* eager_tensor_properties_get_place_str(EagerTensorObject* self,
                                                void* closure) {
  std::stringstream ostr;
  ostr << self->eagertensor.place();
  return ToPyObject(ostr.str());
}

struct PyGetSetDef variable_properties[] = {
    {"shape", (getter)eager_tensor_properties_get_shape, nullptr, nullptr,
     nullptr},
    {"stop_gradient", (getter)eager_tensor_properties_get_stop_gradient,
     (setter)eager_tensor_properties_set_stop_gradient, nullptr, nullptr},
    {"dtype", (getter)eager_tensor_properties_get_dtype, nullptr, nullptr,
     nullptr},
    {"_place_str", (getter)eager_tensor_properties_get_place_str, nullptr,
     nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // namespace pybind
}  // namespace paddle
