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

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include "paddle/fluid/imperative/utils/tensor_utils.h"
#include <stdint.h> // for int8_t
#include <typeindex>
#include <cstdint>
#include <map>
#include <numpy/arrayobject.h>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {
namespace utils {

struct NumpyDataTypeMap {
  std::unordered_map<int, int> proto_to_dtype_;
  std::unordered_map<int, int> dtype_to_proto_;
};

static NumpyDataTypeMap* InitNumpyDataTypeMap();

static NumpyDataTypeMap& gNumpyDataTypeMap() {
  static NumpyDataTypeMap* g_numpy_data_type_map_ = InitNumpyDataTypeMap();
  return *g_numpy_data_type_map_;
}

static inline void RegisterNumpyType(NumpyDataTypeMap* map,
                                     int dtype,
                                     framework::proto::VarType::Type proto_type) {
  map->proto_to_dtype_.emplace(static_cast<int>(proto_type), dtype);
  map->dtype_to_proto_.emplace(dtype, static_cast<int>(proto_type));
}

static NumpyDataTypeMap* InitNumpyDataTypeMap() {
  NumpyDataTypeMap* map = new NumpyDataTypeMap();

  RegisterNumpyType(map, NPY_HALF, framework::proto::VarType::FP16);
  RegisterNumpyType(map, NPY_FLOAT, framework::proto::VarType::FP32);
  RegisterNumpyType(map, NPY_DOUBLE, framework::proto::VarType::FP64);
  RegisterNumpyType(map, NPY_INT32, framework::proto::VarType::INT32);
  RegisterNumpyType(map, NPY_INT64, framework::proto::VarType::INT64);
  RegisterNumpyType(map, NPY_BOOL, framework::proto::VarType::BOOL);
  RegisterNumpyType(map, NPY_LONGLONG, framework::proto::VarType::SIZE_T);
  RegisterNumpyType(map, NPY_INT16, framework::proto::VarType::INT16);
  RegisterNumpyType(map, NPY_UINT8, framework::proto::VarType::UINT8);

  return map;
}

static void CheckTensor(const framework::Tensor& tensor) {
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

static int GetDType(const framework::Tensor& tensor) {
  auto it = gNumpyDataTypeMap().proto_to_dtype_.find(
      static_cast<int>(framework::ToDataType(tensor.type())));
  if (it != gNumpyDataTypeMap().proto_to_dtype_.end()) {
    return it->second;
  }

  PADDLE_THROW("Numpy conversion from tensor with type %s is NOT supported",
               tensor.type().name());
}

static std::type_index GetTensorType(int dtype) {
  auto it = gNumpyDataTypeMap().dtype_to_proto_.find(dtype);
  if (it != gNumpyDataTypeMap().dtype_to_proto_.end()) {
    return framework::ToTypeIndex(static_cast<framework::proto::VarType::Type>(it->second));
  }

  PADDLE_THROW("Numpy conversion from numpy array with type %s is NOT supported",
               dtype);
}


static std::vector<npy_intp> DDimToNumpyShape(const framework::DDim& ddim) {
  std::vector<npy_intp> numpy_shape;
  numpy_shape.reserve(ddim.size());
  for (size_t i = 0u; i != ddim.size(); ++i) {
    numpy_shape.emplace_back(static_cast<npy_intp>(ddim[i]));
  }
  return numpy_shape;
}

static std::vector<int64_t> NumpyShapeToVector(int ndim, npy_intp* numpy_shape) {
  std::vector<int64_t> dims;
  dims.reserve(ndim);
  for (int i = 0; i != ndim; ++i) {
    dims.emplace_back(numpy_shape[i]);
  }

  return dims;
}

// NOTICE(minqiyang): try pybind11 here and decide the way of
// implementing numpy bridge after the benchmark.
PyObject* TensorToNumpy(const framework::Tensor& tensor) {
  // check the place of tensor
  CheckTensor(tensor);

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
  PyObject* array = PyArray_New(
      &PyArray_Type,
      tensor.dims().size(),
      sizes.data(),
      dtype,
      strides.data(),
      tensor.Holder()->ptr(),
      0,
      NPY_ARRAY_BEHAVED,
      nullptr);

  if (!array) {
    return nullptr;
  }

  // TODO(minqiyang): we should avoid to release the tensor here,
  // since we passed it to numpy.
  // // create a VarBase py_object and set it to Numpy PyArrayObj
  // if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(array.get()),
                            // make_var_base(tensor)) == -1) {
    // return nullptr;
  // }

  return array;
}

framework::Tensor TensorFromNumpy(PyObject* obj) {
  // check numpy ndarray
  if (!PyArray_Check(obj)) {
    PADDLE_THROW("Only np.ndarray is supported, got %s here",
                 Py_TYPE(obj)->tp_name);
  }
  auto array = reinterpret_cast<PyArrayObject*>(obj);

  // get the meta data of numpy ndarray
  int ndim = PyArray_NDIM(array);
  std::vector<int64_t> dims = NumpyShapeToVector(ndim, PyArray_DIMS(array));
  // NOTE(minqiyang): after tensor's requested size calculation, we will throw
  // the strides info away, because tensor will calculate strides itself.
  std::vector<int64_t> strides = NumpyShapeToVector(ndim, PyArray_STRIDES(array));

  // calculate the requested size of tensor
  size_t requested_size = 0U;
  if (dims.size() >= 1 && strides.size() >= 1) {
    requested_size = static_cast<size_t>(dims[0] * strides[0]);
  }

  // check byte order
  if (!PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE)) {
    PADDLE_ENFORCE(PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder,
                                           NPY_NATIVE),
                   "numpy array's byte orders should keep the same with the native byte order");
  }

  // get the data ptr and construct the memory allocator attribute
  void* data_ptr = PyArray_DATA(array);
  framework::Tensor tensor;
  tensor.clear();
  tensor.Resize(framework::make_ddim(dims));
  tensor.mutable_data(platform::CPUPlace(), GetTensorType(PyArray_TYPE(array)),
                      memory::Allocator::Attr::kNumpyShared, 0U);
  // NOTE(minqiyang): must be CPUAllocation here, we will down cast this
  // allocation to CPUAllocation
  memory::allocation::CPUAllocation* cpu_allocation = reinterpret_cast<memory::allocation::CPUAllocation*>(tensor.Holder().get());
  cpu_allocation->share_data_with(data_ptr, requested_size);

  Py_INCREF(obj);

  return tensor;
}

}  // namespace utils
}  // namespace imperative
}  // namespace paddle
