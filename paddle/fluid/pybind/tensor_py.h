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
#include <string>
#include <tuple>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace paddle {
namespace pybind {
namespace details {

template <bool less, size_t I, typename... ARGS>
struct CastToPyBufferImpl;

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<false, I, ARGS...> {
  pybind11::buffer_info operator()(const framework::Tensor &tensor) {
    PADDLE_THROW("This type of tensor cannot be expose to Python");
    return pybind11::buffer_info();
  }
};

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<true, I, ARGS...> {
  using CUR_TYPE = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  pybind11::buffer_info operator()(const framework::Tensor &tensor) {
    if (std::type_index(typeid(CUR_TYPE)) == tensor.type()) {
      auto dim_vec = framework::vectorize(tensor.dims());
      std::vector<size_t> dims_outside;
      std::vector<size_t> strides;
      dims_outside.resize(dim_vec.size());
      strides.resize(dim_vec.size());

      size_t prod = 1;
      for (size_t i = dim_vec.size(); i != 0; --i) {
        dims_outside[i - 1] = (size_t)dim_vec[i - 1];
        strides[i - 1] = sizeof(CUR_TYPE) * prod;
        prod *= dims_outside[i - 1];
      }
      framework::Tensor dst_tensor;
      if (paddle::platform::is_gpu_place(tensor.place())) {
#ifdef PADDLE_WITH_CUDA
        auto *src_ptr = static_cast<const void *>(tensor.data<CUR_TYPE>());
        auto *dst_ptr = static_cast<void *>(dst_tensor.mutable_data<CUR_TYPE>(
            tensor.dims(), platform::CPUPlace()));

        paddle::platform::GpuMemcpySync(dst_ptr, src_ptr,
                                        sizeof(CUR_TYPE) * tensor.numel(),
                                        cudaMemcpyDeviceToHost);
#else
        PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
      } else if (paddle::platform::is_cpu_place(tensor.place())) {
        dst_tensor = tensor;
      }

      if (std::type_index(typeid(CUR_TYPE)) ==
          std::type_index(typeid(platform::float16))) {
        return pybind11::buffer_info(
            dst_tensor.data<CUR_TYPE>(), sizeof(CUR_TYPE),
            "e", /* np.dtype('e') == np.float16 */
            (size_t)framework::arity(dst_tensor.dims()), dims_outside, strides);
      } else {
        return pybind11::buffer_info(
            dst_tensor.data<CUR_TYPE>(), sizeof(CUR_TYPE),
            pybind11::format_descriptor<CUR_TYPE>::format(),
            (size_t)framework::arity(dst_tensor.dims()), dims_outside, strides);
      }
    } else {
      constexpr bool less = I + 1 < std::tuple_size<std::tuple<ARGS...>>::value;
      return CastToPyBufferImpl<less, I + 1, ARGS...>()(tensor);
    }
  }
};

}  // namespace details

inline pybind11::buffer_info CastToPyBuffer(const framework::Tensor &tensor) {
  auto buffer_info =
      details::CastToPyBufferImpl<true, 0, float, int, double, int64_t, bool,
                                  uint8_t, platform::float16>()(tensor);
  return buffer_info;
}

template <typename T>
T TensorGetElement(const framework::Tensor &self, size_t offset) {
  if (platform::is_cpu_place(self.place())) {
    return self.data<T>()[offset];
  } else {
    std::shared_ptr<framework::Tensor> dst(new framework::Tensor);
    framework::TensorCopySync(self, platform::CPUPlace(), dst.get());
    return dst->data<T>()[offset];
  }
}

// TODO(dzhwinter) : fix the redundent Tensor allocate and free
template <typename T>
void TensorSetElement(framework::Tensor *self, size_t offset, T elem) {
  if (platform::is_gpu_place(self->place())) {
    std::shared_ptr<framework::Tensor> dst(new framework::Tensor);
    framework::TensorCopySync(*self, platform::CPUPlace(), dst.get());
    dst->data<T>()[offset] = elem;
    framework::TensorCopySync(*dst.get(), self->place(), self);

  } else if (platform::is_cpu_place(self->place())) {
    self->data<T>()[offset] = elem;
  }
}

template <typename T>
void PyCPUTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CPUPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<T>(place);
  std::memcpy(dst, array.data(), sizeof(T) * array.size());
}

template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
void PyCPUTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<uint16_t,
                      pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CPUPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<platform::float16>(place);
  std::memcpy(dst, array.data(), sizeof(uint16_t) * array.size());
}

#ifdef PADDLE_WITH_CUDA
template <typename T>
void PyCUDATensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CUDAPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<T>(place);
  paddle::platform::GpuMemcpySync(dst, array.data(), sizeof(T) * array.size(),
                                  cudaMemcpyHostToDevice);
}

template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
void PyCUDATensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<uint16_t,
                      pybind11::array::c_style | pybind11::array::forcecast>
        array,
    paddle::platform::CUDAPlace place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<platform::float16>(place);
  paddle::platform::GpuMemcpySync(dst, array.data(),
                                  sizeof(uint16_t) * array.size(),
                                  cudaMemcpyHostToDevice);
}

template <typename T>
void PyCUDAPinnedTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        array,
    const paddle::platform::CUDAPinnedPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<T>(place);
  std::memcpy(dst, array.data(), sizeof(T) * array.size());
}

template <>
// This following specialization maps uint16_t in the parameter type to
// platform::float16.
void PyCUDAPinnedTensorSetFromArray(
    framework::Tensor *self,
    pybind11::array_t<uint16_t,
                      pybind11::array::c_style | pybind11::array::forcecast>
        array,
    const paddle::platform::CUDAPinnedPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }

  self->Resize(framework::make_ddim(dims));
  auto *dst = self->mutable_data<platform::float16>(place);
  std::memcpy(dst, array.data(), sizeof(uint16_t) * array.size());
}
#endif

}  // namespace pybind
}  // namespace paddle
