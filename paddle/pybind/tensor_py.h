/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <string>
#include "paddle/framework/tensor.h"
#include "paddle/memory/memcpy.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace paddle {

namespace pybind {

namespace details {

template <bool less, size_t I, typename... ARGS>
struct CastToPyBufferImpl;

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<false, I, ARGS...> {
  py::buffer_info operator()(framework::Tensor &tensor) {
    PADDLE_THROW("This type of tensor cannot be expose to Python");
    return py::buffer_info();
  }
};

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<true, I, ARGS...> {
  using CUR_TYPE = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  py::buffer_info operator()(framework::Tensor &tensor) {
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
        // TODO(qijun): Here we use default CUDA stream to set GPU Tensor to
        // a Python numpy array. It's better to manage CDUA stream unifiedly.
        paddle::platform::GpuMemcpySync(dst_ptr, src_ptr,
                                        sizeof(CUR_TYPE) * tensor.numel(),
                                        cudaMemcpyDeviceToHost);
#else
        PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
      } else if (paddle::platform::is_cpu_place(tensor.place())) {
        dst_tensor = tensor;
      }
      return py::buffer_info(
          dst_tensor.mutable_data<CUR_TYPE>(dst_tensor.place()),
          sizeof(CUR_TYPE), py::format_descriptor<CUR_TYPE>::format(),
          (size_t)framework::arity(dst_tensor.dims()), dims_outside, strides);
    } else {
      constexpr bool less = I + 1 < std::tuple_size<std::tuple<ARGS...>>::value;
      return CastToPyBufferImpl<less, I + 1, ARGS...>()(tensor);
    }
  }
};
}  // namespace details
inline py::buffer_info CastToPyBuffer(framework::Tensor &tensor) {
  auto buffer_info =
      details::CastToPyBufferImpl<true, 0, float, int, double, int64_t, bool>()(
          tensor);
  return buffer_info;
}

template <typename T>
T TensorGetElement(framework::Tensor &self, size_t offset) {
  PADDLE_ENFORCE(platform::is_cpu_place(self.place()));
  return self.data<T>()[offset];
}

template <typename T>
void TensorSetElement(framework::Tensor &self, size_t offset, T elem) {
  PADDLE_ENFORCE(platform::is_cpu_place(self.place()));
  self.data<T>()[offset] = elem;
}

template <typename T>
void PyCPUTensorSetFromArray(
    framework::Tensor &self,
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    paddle::platform::CPUPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back((int)array.shape()[i]);
  }

  self.Resize(framework::make_ddim(dims));
  auto *dst = self.mutable_data<T>(place);
  std::memcpy(dst, array.data(), sizeof(T) * array.size());
}

#ifdef PADDLE_WITH_CUDA
template <typename T>
void PyCUDATensorSetFromArray(
    framework::Tensor &self,
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    paddle::platform::GPUPlace &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back((int)array.shape()[i]);
  }

  self.Resize(framework::make_ddim(dims));
  auto *dst = self.mutable_data<T>(place);
  // TODO(qijun): Here we use default CUDA stream to set a Python numpy
  // array to a GPU Tensor. It's better to manage CDUA stream unifiedly.
  paddle::platform::GpuMemcpySync(dst, array.data(), sizeof(T) * array.size(),
                                  cudaMemcpyHostToDevice);
}
#endif

}  // namespace pybind
}  // namespace paddle
