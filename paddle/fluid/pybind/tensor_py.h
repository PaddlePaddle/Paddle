/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <Python.h>
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;

// Note: Since float16 is not a builtin type in C++, we register
// paddle::platform::float16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
template <>
struct npy_format_descriptor<paddle::platform::float16> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16_);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "e" represents float16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "e";
  }
  static PYBIND11_DESCR name() { return _("float16"); }
};

}  // namespace detail
}  // namespace pybind11

namespace paddle {
namespace pybind {

namespace details {

template <typename T>
class PYBIND11_HIDDEN NumpyAllocation : public memory::Allocation {
 public:
  explicit NumpyAllocation(const py::array &arr)
      : Allocation(const_cast<void *>(arr.data()), sizeof(T) * (arr.size()),
                   paddle::platform::CPUPlace()),
        arr_(arr.ptr()) {
    PADDLE_ENFORCE_NOT_NULL(arr_, platform::errors::InvalidArgument(
                                      "The underlying PyObject pointer of "
                                      "numpy array cannot be nullptr"));
    PADDLE_ENFORCE_NE(
        arr_, Py_None,
        platform::errors::PreconditionNotMet(
            "The underlying PyObject pointer of numpy array cannot be None"));
    Py_INCREF(arr_);
  }
  ~NumpyAllocation() override {
    py::gil_scoped_acquire gil;
    Py_DECREF(arr_);
  }

 private:
  PyObject *arr_;
};

template <typename T>
struct ValidDTypeToPyArrayChecker {
  static constexpr bool kValue = false;
};

#define DECLARE_VALID_DTYPE_TO_PY_ARRAY(type) \
  template <>                                 \
  struct ValidDTypeToPyArrayChecker<type> {   \
    static constexpr bool kValue = true;      \
  }

DECLARE_VALID_DTYPE_TO_PY_ARRAY(platform::float16);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(float);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(double);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(bool);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int8_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int16_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int64_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(uint8_t);

inline std::string TensorDTypeToPyDTypeStr(
    framework::proto::VarType::Type type) {
#define TENSOR_DTYPE_TO_PY_DTYPE(T, proto_type)                             \
  if (type == proto_type) {                                                 \
    if (std::is_same<T, platform::float16>::value) {                        \
      return "e";                                                           \
    } else {                                                                \
      constexpr auto kIsValidDType = ValidDTypeToPyArrayChecker<T>::kValue; \
      PADDLE_ENFORCE_EQ(kIsValidDType, true,                                \
                        "This type of tensor cannot be expose to Python");  \
      return py::format_descriptor<T>::format();                            \
    }                                                                       \
  }

  _ForEachDataType_(TENSOR_DTYPE_TO_PY_DTYPE);
#undef TENSOR_DTYPE_TO_PY_DTYPE
  PADDLE_THROW("Unsupported data type %d", static_cast<int>(type));
}

}  // namespace details

template <typename T>
T TensorGetElement(const framework::Tensor &self, size_t offset) {
  PADDLE_ENFORCE_LT(offset, self.numel());
  T b = static_cast<T>(0);
  if (platform::is_cpu_place(self.place())) {
    b = self.data<T>()[offset];
#ifdef PADDLE_WITH_CUDA
  } else {
    const T *a = self.data<T>();
    auto p = BOOST_GET_CONST(platform::CUDAPlace, self.place());
    paddle::memory::Copy(platform::CPUPlace(), &b, p, a + offset, sizeof(T),
                         nullptr);
#endif
  }
  return b;
}

template <typename T>
void TensorSetElement(framework::Tensor *self, size_t offset, T elem) {
  PADDLE_ENFORCE_LT(offset, self->numel());
  if (platform::is_cpu_place(self->place())) {
    self->mutable_data<T>(self->place())[offset] = elem;
#ifdef PADDLE_WITH_CUDA
  } else {
    auto p = BOOST_GET_CONST(platform::CUDAPlace, self->place());
    T *a = self->mutable_data<T>(p);
    paddle::memory::Copy(p, a + offset, platform::CPUPlace(), &elem, sizeof(T),
                         nullptr);
#endif
  }
}

template <typename T, typename P>
void SetTensorFromPyArrayT(
    framework::Tensor *self,
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    const P &place, bool zero_copy) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }
  self->Resize(framework::make_ddim(dims));

  if (paddle::platform::is_cpu_place(place)) {
    if (zero_copy) {
      auto holder = std::make_shared<details::NumpyAllocation<T>>(array);
      auto type = framework::ToDataType(std::type_index(typeid(T)));
      self->ResetHolderWithType(holder, type);
    } else {
      auto dst = self->mutable_data<T>(place);
      std::memcpy(dst, array.data(), array.nbytes());
    }
  } else {
#ifdef PADDLE_WITH_CUDA
    auto dst = self->mutable_data<T>(place);
    if (paddle::platform::is_cuda_pinned_place(place)) {
      std::memcpy(dst, array.data(), array.nbytes());
    } else if (paddle::platform::is_gpu_place(place)) {
      paddle::platform::GpuMemcpySync(dst, array.data(), array.nbytes(),
                                      cudaMemcpyHostToDevice);
    } else {
      PADDLE_THROW(
          "Incompatible place type: Tensor.set() supports CPUPlace, CUDAPlace "
          "and CUDAPinnedPlace, but got %s!",
          place);
    }
#else
    PADDLE_THROW("Not supported GPU, please compile WITH_GPU option");
#endif
  }
}

template <typename P>
void SetTensorFromPyArray(framework::Tensor *self, const py::object &obj,
                          const P &place, bool zero_copy) {
  auto array = obj.cast<py::array>();
  if (py::isinstance<py::array_t<float>>(array)) {
    SetTensorFromPyArrayT<float, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int>>(array)) {
    SetTensorFromPyArrayT<int, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int64_t>>(array)) {
    SetTensorFromPyArrayT<int64_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<double>>(array)) {
    SetTensorFromPyArrayT<double, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int8_t>>(array)) {
    SetTensorFromPyArrayT<int8_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int16_t>>(array)) {
    SetTensorFromPyArrayT<int16_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<uint8_t>>(array)) {
    SetTensorFromPyArrayT<uint8_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<paddle::platform::float16>>(array)) {
    SetTensorFromPyArrayT<paddle::platform::float16, P>(self, array, place,
                                                        zero_copy);
  } else if (py::isinstance<py::array_t<uint16_t>>(array)) {
    // TODO(cql): temporary keeping uint16, which is used for casting float16
    // before. It should be depracated later.
    SetTensorFromPyArrayT<paddle::platform::float16, P>(self, array, place,
                                                        zero_copy);
  } else if (py::isinstance<py::array_t<bool>>(array)) {
    SetTensorFromPyArrayT<bool, P>(self, array, place, zero_copy);
  } else {
    PADDLE_THROW(
        "Incompatible data or style type: tensor.set() supports bool, float16, "
        "float32, "
        "float64, "
        "int8, int16, int32, int64 and uint8, uint16, but got %s!",
        array.dtype());
  }
}

template <typename T, size_t D>
void _sliceCompute(const framework::Tensor *in, framework::Tensor *out,
                   const platform::CPUDeviceContext &ctx,
                   const std::vector<int> &axes,
                   const std::vector<int> &starts) {
  auto &eigen_place = *ctx.eigen_device();
  auto place = in->place();
  auto out_dims = out->dims();
  auto in_dims = in->dims();

  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = out_dims[i];
  }
  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *in);
  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out);
  out_t.device(eigen_place) = in_t.slice(offsets, extents);
}

template <typename T>
void _concatCompute(const std::vector<paddle::framework::Tensor> &ins,
                    paddle::framework::Tensor *out,
                    const platform::CPUDeviceContext &ctx, int64_t axis) {
  if (axis == 0 && ins.size() < 10) {
    size_t output_offset = 0;
    for (auto &in : ins) {
      auto in_stride = framework::stride_numel(in.dims());
      auto out_stride = framework::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          ctx, axis, out->data<T>() + output_offset, out_stride, in.data<T>(),
          in_stride, in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, T>
        concat_functor;
    concat_functor(ctx, ins, static_cast<int>(axis), out);
  }
}

inline void _getSliceinfo(const framework::Tensor &self, py::object obj,
                          const int64_t dim, int64_t *pstart, int64_t *pstop,
                          int64_t *pstep, int64_t *pslicelength) {
  auto &start = *pstart;
  auto &stop = *pstop;
  auto &step = *pstep;
  auto &slicelength = *pslicelength;
  const framework::DDim &srcDDim = self.dims();
  if (dim < 0 || dim >= srcDDim.size()) {
    throw py::index_error();
  }
  if (py::isinstance<py::slice>(obj)) {
    size_t lstart, lstop, lstep, lslicelength;
    py::slice s = static_cast<py::slice>(obj);
    if (!s.compute(srcDDim[dim], &lstart, &lstop, &lstep, &lslicelength)) {
      throw py::index_error();
    }
    start = static_cast<int64_t>(lstart);
    stop = static_cast<int64_t>(lstop);
    step = static_cast<int64_t>(lstep);
    slicelength = static_cast<int64_t>(lslicelength);
  } else if (py::isinstance<py::int_>(obj)) {
    start = static_cast<int64_t>(static_cast<py::int_>(obj));
    if (std::abs(start) >= srcDDim[dim]) {
      throw py::index_error();
    }
    start = (start >= 0) ? start : srcDDim[dim] - start;
    stop = start + 1;
    step = 1;
    slicelength = 1;
  } else {
    throw py::index_error();
  }
}

inline framework::Tensor *_getTensor(const framework::Tensor &self,
                                     const framework::DDim &ddim) {
  framework::Tensor *output = new framework::Tensor();
  output->Resize(ddim);
  auto place = self.place();
  if (platform::is_cpu_place(place)) {
    output->mutable_data(BOOST_GET_CONST(platform::CPUPlace, place),
                         self.type());
#ifdef PADDLE_WITH_CUDA
  } else {
    if (platform::is_cuda_pinned_place(place)) {
      output->mutable_data(BOOST_GET_CONST(platform::CUDAPinnedPlace, place),
                           self.type());
    } else if ((platform::is_gpu_place(place))) {
      output->mutable_data(BOOST_GET_CONST(platform::CUDAPlace, place),
                           self.type());
    }
#endif
  }
  return output;
}

template <typename T>
void _sliceDapper(const framework::Tensor *in, framework::Tensor *out,
                  const platform::CPUDeviceContext &ctx,
                  const std::vector<int> &axes, const std::vector<int> &starts,
                  int size) {
  switch (size) {
    case 1:
      _sliceCompute<T, 1>(in, out, ctx, axes, starts);
      break;
    case 2:
      _sliceCompute<T, 2>(in, out, ctx, axes, starts);
      break;
    case 3:
      _sliceCompute<T, 3>(in, out, ctx, axes, starts);
      break;
    case 4:
      _sliceCompute<T, 4>(in, out, ctx, axes, starts);
      break;
    case 5:
      _sliceCompute<T, 5>(in, out, ctx, axes, starts);
      break;
    case 6:
      _sliceCompute<T, 6>(in, out, ctx, axes, starts);
      break;
    case 7:
      _sliceCompute<T, 7>(in, out, ctx, axes, starts);
      break;
    case 8:
      _sliceCompute<T, 8>(in, out, ctx, axes, starts);
      break;
    case 9:
      _sliceCompute<T, 9>(in, out, ctx, axes, starts);
      break;
    default:
      PADDLE_THROW("dim size not exepected, current is %d", size);
      break;
  }
}

template <typename T>
inline framework::Tensor *_sliceWrapper(const framework::Tensor &self,
                                        const platform::CPUDeviceContext &ctx,
                                        py::object obj, int dim, int64_t start,
                                        int64_t slicelength) {
  framework::DDim dstDDim = self.dims();
  dstDDim[dim] = static_cast<int64_t>(slicelength);
  std::vector<int> axes({dim});
  std::vector<int> starts({static_cast<int>(start)});
  framework::Tensor *output = _getTensor(self, dstDDim);
  _sliceDapper<T>(&self, output, ctx, axes, starts, dstDDim.size());
  return output;
}

template <typename T>
inline framework::Tensor *_sliceAndConcat(const framework::Tensor &self,
                                          py::object obj, int dim) {
  platform::CPUDeviceContext ctx;
  int64_t start, stop, step, slicelength;
  _getSliceinfo(self, obj, dim, &start, &stop, &step, &slicelength);
  if (step == 1 || slicelength == 1) {
    return _sliceWrapper<T>(self, ctx, obj, dim, start, slicelength);
  } else {
    std::vector<framework::Tensor> ins;
    for (auto i = 0; i < slicelength; ++i, start += step) {
      ins.emplace_back(*_sliceWrapper<T>(self, ctx, obj, dim, start, 1));
    }

    // do the concat operation
    framework::DDim dstDDim = self.dims();
    dstDDim[dim] = static_cast<int64_t>(slicelength);
    framework::Tensor *output1 = _getTensor(self, dstDDim);
    _concatCompute<T>(ins, output1, ctx, dim);
    return output1;
  }
}

inline framework::Tensor *_sliceTensor(const framework::Tensor &self,
                                       py::object obj, int dim) {
  auto src_type = self.type();
  switch (src_type) {
    case framework::proto::VarType::FP16:
      return _sliceAndConcat<paddle::platform::float16>(self, obj, dim);
    case framework::proto::VarType::FP32:
      return _sliceAndConcat<float>(self, obj, dim);
    case framework::proto::VarType::FP64:
      return _sliceAndConcat<double>(self, obj, dim);
    case framework::proto::VarType::INT8:
      return _sliceAndConcat<int8_t>(self, obj, dim);
    case framework::proto::VarType::INT16:
      return _sliceAndConcat<int16_t>(self, obj, dim);
    case framework::proto::VarType::INT32:
      return _sliceAndConcat<int>(self, obj, dim);
    case framework::proto::VarType::INT64:
      return _sliceAndConcat<int64_t>(self, obj, dim);
    case framework::proto::VarType::BOOL:
      return _sliceAndConcat<bool>(self, obj, dim);
    case framework::proto::VarType::UINT8:
      return _sliceAndConcat<uint8_t>(self, obj, dim);
    default:
      PADDLE_THROW("Not support type %d", src_type);
  }
}

inline framework::Tensor *_pySliceTensor(const framework::Tensor &self,
                                         py::object obj) {
  if (py::isinstance<py::tuple>(obj)) {
    py::list l = static_cast<py::list>(obj);
    std::unique_ptr<framework::Tensor> target;
    framework::Tensor *src = const_cast<framework::Tensor *>(&self);
    for (auto i = 0; i < static_cast<int>(l.size()); ++i) {
      src = _sliceTensor(*src, l[i], i);
      if (i + 1 == static_cast<int>(l.size())) {
        return src;
      } else {
        target.reset(src);
      }
    }
    return nullptr;
  } else {
    return _sliceTensor(self, obj, 0);
  }
}

inline framework::Tensor *PySliceTensor(const framework::Tensor &self,
                                        py::object obj) {
  if (platform::is_gpu_place(self.place())) {
    std::unique_ptr<framework::Tensor> holder;
    framework::Tensor src;
    framework::TensorCopySync(self, platform::CPUPlace(), &src);
    framework::Tensor *output = _pySliceTensor(src, obj);
    holder.reset(output);
    framework::Tensor *dst = _getTensor(*output, output->dims());
    framework::TensorCopySync(*output, self.place(), dst);
    return dst;
  } else {
    return _pySliceTensor(self, obj);
  }
}

inline py::array TensorToPyArray(const framework::Tensor &tensor,
                                 bool need_deep_copy = false) {
  if (!tensor.IsInitialized()) {
    return py::array();
  }
  bool is_gpu_tensor = platform::is_gpu_place(tensor.place());
  const auto &tensor_dims = tensor.dims();
  auto tensor_dtype = tensor.type();
  size_t sizeof_dtype = framework::SizeOfType(tensor_dtype);

  std::vector<size_t> py_dims(tensor_dims.size());
  std::vector<size_t> py_strides(tensor_dims.size());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    py_dims[i] = (size_t)tensor_dims[i];
    py_strides[i] = sizeof_dtype * numel;
    numel *= py_dims[i];
  }

  const void *tensor_buf_ptr = tensor.data<void>();

  std::string py_dtype_str = details::TensorDTypeToPyDTypeStr(tensor.type());

  if (!is_gpu_tensor) {
    if (!need_deep_copy) {
      return py::array(py::buffer_info(
          const_cast<void *>(tensor_buf_ptr), sizeof_dtype, py_dtype_str,
          static_cast<size_t>(tensor.dims().size()), py_dims, py_strides));
    } else {
      py::array py_arr(py::dtype(py_dtype_str.c_str()), py_dims, py_strides);
      PADDLE_ENFORCE_EQ(py_arr.writeable(), true,
                        platform::errors::InvalidArgument(
                            "PyArray must be writable, otherwise memory leak "
                            "or double free would occur"));
      PADDLE_ENFORCE_EQ(py_arr.owndata(), true,
                        platform::errors::InvalidArgument(
                            "PyArray must own data, otherwise memory leak "
                            "or double free would occur"));
      platform::CPUPlace place;
      size_t copy_bytes = sizeof_dtype * numel;
      paddle::memory::Copy(place, py_arr.mutable_data(), place, tensor_buf_ptr,
                           copy_bytes);
      return py_arr;
    }
  }

#ifdef PADDLE_WITH_CUDA
  py::array py_arr(py::dtype(py_dtype_str.c_str()), py_dims, py_strides);
  PADDLE_ENFORCE(py_arr.writeable() && py_arr.owndata(),
                 "PyArray must be writable and own data, otherwise memory leak "
                 "or double free would occur");

  size_t copy_bytes = sizeof_dtype * numel;
  paddle::platform::GpuMemcpySync(py_arr.mutable_data(), tensor_buf_ptr,
                                  copy_bytes, cudaMemcpyDeviceToHost);
  return py_arr;
#else
  PADDLE_THROW("CUDAPlace is not supported when not compiled with CUDA");
#endif
}

}  // namespace pybind
}  // namespace paddle
