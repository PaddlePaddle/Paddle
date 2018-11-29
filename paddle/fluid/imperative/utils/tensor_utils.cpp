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

#include "paddle/fluid/imperative/utils/tensor_utils.h"

#include <Python.h>

#ifdef WITH_PYTHON
#include <numpy/arrayobject.h>
#endif

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {
namespace utils {

namespace {

static void CheckTensor(const Tensor& tensor) {
  // NOTE(minqiyang): add API hint when any check meets error here
  // check if tensor is initialized
  PADDLE_ENFORCE(tensor.IsInitialized(),
                 "Tensor should be initialized first");

  // check the place of tensor
  // NOTE(minqiyang): we could only support numpy bridge
  // between cpu tensor and ndarray now.
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(tensor.place()),
                 "Can only convert CPU tensor to numpy.");
}

static int GetDType(const Tensor& tensor) {
  proto::VarType::Type type = ToDataType(tensor.type());
  switch (type) {
    case proto::VarType::FP16:
      return NPY_HALF;
    case proto::VarType::FP32:
      return NPY_FLOAT;
    case proto::VarType::FP64:
      return NPY_DOUBLE;
    case proto::VarType::INT32:
      return NPY_INT32;
    case proto::VarType::INT64:
      return NPY_INT64;
    case proto::VarType::SIZE_T:
      return NPY_LONGLONG;
    case proto::VarType::INT16:
      return NPY_INT16;
    case proto::VarType::UINT8:
      return NPY_UINT8;
    default:
      PADDLE_THROW("Numpy conversion from tensor with type %s is NOT supported",
                   type.name());
  }
}

static std::vector<npy_intp> DDimToNumpyShape(const framework::DDim& ddim) {
  std::vector<npy_intp> numpy_shape;
  numpy_shape.reserve(ddim.size());
  for (size_t i = 0u; i != ddim.size(); ++i) {
    numpy_shape.emplace_back(static_cast<npy_intp>(ddim[i]));
  }
}

} // NOLINT

// NOTICE(minqiyang): try pybind11 here and decide the way of
// implementing numpy bridge after the benchmark.
void TensorToNumpy(const framework::Tensor& tensor) {
  // check the place of tensor
  CheckTensor();

  // get meta data (type, shape e.g..) from tensor
  auto dtype = GetDType(tensor);
  auto sizes = DDimToNumpyShape(tensor.dims());
  // strides should take the sizeof type in count
  auto strides = DDimToNumpyShape(framework::stride(tensor.dims()));
  size_t type_size = framework::SizeOfType(tensor.type());
  for (auto& stride : strides) {
    stride *= type_size;
  }

  // create numpy array
  auto array = THPObjectPtr(PyArray_New(
      &PyArray_Type,
      tensor.dims().size(),
      sizes.data(),
      dtype,
      strides.data(),
      tensor.mutable_data(),
      0,
      NPY_ARRAY_BEHAVED,
      nullptr));

  if (!array) {
    return nullptr;
  }

  // create a VarBase py_object and set it to Numpy PyArrayObj
  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(array.get()),
                            make_var_base(tensor)) == -1) {
    // TODO(minqiyang): we should avoid to release the tensor here,
    // since we passed it to numpy.
    return nullptr;
  }

  return array.release();
}

framework::Tensor TensorFromNumpy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    PADDLE_THROW("Only np.ndarray is supported, got %s here",
                 Py_TYPE(obj)->tp_name);
  }

  auto array = reinterpret_cast<PyArrayObject*>(obj);
  // get the number of dim
  int ndim = PyArray_NDIM(array);
  auto sizes = to_aten_shape(ndim, PyArray_DIMS(array));
  auto strides = to_aten_shape(ndim, PyArray_STRIDES(array));

  int item_size = PyArray_ITEMSIZE(array);
  for (auto& stride : strides) {
    PADDLE_ENFORCE(stride >= 0,
                   "negative strides in numpy array is NOT supported now");
    PADDLE_ENFORCE(stride % item_size == 0,
                   "numpy array's strides is NOT the times of item size"
                   "please copy the numpy array and redo this method");
    stride /= item_size;
  }

  // check byte order
  if (!PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE)) {
    PADDLE_ENFORCE(PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder,
                                           NPY_NATIVE),
                   "numpy array's byte orders should keep the same with the native byte order");
  }

  void* data_ptr = PyArray_DATA(array);
  Tensor tensor;
  tensor.mutable_data();

  Py_INCREF(obj);

  return tensor;
}

}  // namespace utils
}  // namespace imperative
}  // namespace paddle

