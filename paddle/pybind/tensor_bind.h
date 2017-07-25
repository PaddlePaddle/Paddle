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
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(tensor.holder_->place()),
                   "Only CPU tensor can cast to numpy array");

    if (std::type_index(typeid(CUR_TYPE)) == tensor.holder_->type()) {
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
      Tensor dst_tensor;
      if (paddle::platform::is_gpu_place(tensor.holder_->place())) {
        dst_tensor.CopyFrom(tensor, platform::CPUPlace());
      } else if (paddle::platform::is_gpu_place(tensor.holder_->place())) {
        dst_tensor = tensor;
      }
      return py::buffer_info(
          dst_tensor.mutable_data<CUR_TYPE>(dst_tensor.holder_->place()),
          sizeof(CUR_TYPE),
          py::format_descriptor<CUR_TYPE>::format(),
          (size_t)framework::arity(dst_tensor.dims()),
          dims_outside,
          strides);
    } else {
      constexpr bool less = I + 1 < std::tuple_size<std::tuple<ARGS...>>::value;
      return CastToPyBufferImpl<less, I + 1, ARGS...>()(tensor);
    }
  }
};
}  // namespace details
inline py::buffer_info CastToPyBuffer(framework::Tensor &tensor) {
  auto buffer_info = details::CastToPyBufferImpl<true, 0, float, int>()(tensor);
  return buffer_info;
}

template <typename T>
void PyTensorSetFromArray(
    framework::Tensor &self,
    py::array_t<T, py::array::c_style | py::array::forcecast> array) {
  std::vector<int> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back((int)array.shape()[i]);
  }

  self.Resize(framework::make_ddim(dims));
  auto *dst = self.mutable_data<T>(self.place());

  if (paddle::platform::is_cpu_place(self.place())) {
    paddle::memory::Copy<paddle::platform::CPUPlace,
                         paddle::platform::CPUPlace>(
        place, dst, place, array.data(), sizeof(T) * array.size());
  } else if (paddle::platform::is_gpu_place(place)) {
#ifdef PADDLE_ONLY_CPU
    PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#else
    paddle::memory::Copy<paddle::platform::GPUPlace,
                         paddle::platform::CPUPlace>(
        place,
        dst,
        paddle::platform::CPUPlace(),
        array.data(),
        sizeof(T) * array.size());
#endif
  }
}

}  // namespace pybind
}  // namespace paddle
