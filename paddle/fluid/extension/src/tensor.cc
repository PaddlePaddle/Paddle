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

#include "paddle/fluid/extension/include/tensor.h"
#include <utility>
#include "paddle/fluid/framework/custom_tensor_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {

#define GET_CASTED_TENSOR                               \
  if (!tensor_) {                                       \
    tensor_ = std::make_shared<framework::LoDTensor>(); \
  }                                                     \
  auto *tensor = static_cast<framework::LoDTensor *>(tensor_.get());

void Tensor::reshape(const std::vector<int> &shape) {
  GET_CASTED_TENSOR
  tensor->Resize(framework::make_ddim(shape));
}

Tensor::Tensor(const PlaceType &place)
    : tensor_(std::make_shared<framework::LoDTensor>()), place_(place) {}

template <typename T>
T *Tensor::mutable_data(const PlaceType &place) {
  place_ = place;
  return mutable_data<T>();
}

template <typename T>
T *Tensor::mutable_data() {
  GET_CASTED_TENSOR
  PADDLE_ENFORCE_GT(
      tensor->numel(), 0,
      platform::errors::PreconditionNotMet(
          "You should call Tensor::Reshape(const std::vector<int> "
          "&shape)"
          "function before retrieving mutable_data from input tensor."));
  switch (static_cast<int>(place_)) {
    case static_cast<int>(PlaceType::kCPU): {
      return tensor->mutable_data<T>(platform::CPUPlace());
    }
    case static_cast<int>(PlaceType::kGPU): {
#ifdef PADDLE_WITH_CUDA
      int device_num = platform::GetCurrentDeviceId();
      return tensor->mutable_data<T>(platform::CUDAPlace(device_num));
#endif
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "CustomOp unsupported place: %d", static_cast<int>(place_)));
  }
}

template <typename T>
T *Tensor::data() const {
  GET_CASTED_TENSOR;
  auto *res = tensor->data<T>();
  return res;
}

DataType Tensor::type() const {
  GET_CASTED_TENSOR;
  auto type = tensor->type();
  if (type == framework::proto::VarType::FP32) {
    return DataType::FLOAT32;
  } else if (type == framework::proto::VarType::INT64) {
    return DataType::INT64;
  } else if (type == framework::proto::VarType::INT32) {
    return DataType::INT32;
  } else if (type == framework::proto::VarType::INT16) {
    return DataType::INT16;
  } else if (type == framework::proto::VarType::INT8) {
    return DataType::INT8;
  } else if (type == framework::proto::VarType::UINT8) {
    return DataType::UINT8;
  } else if (type == framework::proto::VarType::FP64) {
    return DataType::FLOAT64;
  } else if (type == framework::proto::VarType::BF16) {
    return DataType::BFLOAT16;
  } else if (type == framework::proto::VarType::FP16) {
    return DataType::FLOAT16;
  } else if (type == framework::proto::VarType::COMPLEX64) {
    return DataType::COMPLEX64;
  } else if (type == framework::proto::VarType::COMPLEX128) {
    return DataType::COMPLEX128;
  }
  return DataType::FLOAT32;
}

template <typename T>
Tensor Tensor::copy_to_gpu() {
#ifdef PADDLE_WITH_CUDA
  GET_CASTED_TENSOR;
  PADDLE_ENFORCE_GE(tensor->numel(), 0,
                    platform::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);
  Tensor target = Tensor(PlaceType::kGPU);
  target.reshape(shape());
  auto *p_target_data = target.template mutable_data<T>();
  auto p_src_data = tensor->data<T>();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  int device_num = platform::GetCurrentDeviceId();
  platform::CUDAPlace gpu_place(device_num);
  auto *dev_ctx =
      static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
  if (platform::is_cpu_place(tensor->place())) {
    memory::Copy(gpu_place, static_cast<void *>(p_target_data),
                 platform::CPUPlace(), p_src_data, ele_size, dev_ctx->stream());
  } else {
    memory::Copy(gpu_place, static_cast<void *>(p_target_data), gpu_place,
                 p_src_data, ele_size, dev_ctx->stream());
  }
  cudaStreamSynchronize(dev_ctx->stream());
  return target;
#else
  PADDLE_THROW(
      platform::errors::Unavailable("PaddlePaddle is not compiled with CUDA"));
#endif
  return Tensor(PlaceType::kGPU);
}

template <typename T>
Tensor Tensor::copy_to_cpu() {
  GET_CASTED_TENSOR;
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();
  Tensor target = Tensor(PlaceType::kCPU);
  target.reshape(shape());
  auto *p_target_data = target.template mutable_data<T>();
  if (platform::is_cpu_place(t_place)) {
    std::memcpy(static_cast<void *>(p_target_data), t_data,
                ele_num * sizeof(T));
  } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, t_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
    memory::Copy(platform::CPUPlace(), static_cast<void *>(p_target_data),
                 gpu_place, t_data, ele_num * sizeof(T), dev_ctx->stream());

    cudaStreamSynchronize(dev_ctx->stream());
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle is not compiled with CUDA."));
#endif
  }
  return target;
}

template Tensor Tensor::copy_to_gpu<paddle::platform::float16>();
template Tensor Tensor::copy_to_gpu<paddle::platform::bfloat16>();
template Tensor Tensor::copy_to_gpu<paddle::platform::complex64>();
template Tensor Tensor::copy_to_gpu<paddle::platform::complex128>();
template Tensor Tensor::copy_to_gpu<float>();
template Tensor Tensor::copy_to_gpu<double>();
template Tensor Tensor::copy_to_gpu<int64_t>();
template Tensor Tensor::copy_to_gpu<int32_t>();
template Tensor Tensor::copy_to_gpu<uint8_t>();
template Tensor Tensor::copy_to_gpu<int8_t>();
template Tensor Tensor::copy_to_gpu<int16_t>();

template Tensor Tensor::copy_to_cpu<paddle::platform::float16>();
template Tensor Tensor::copy_to_cpu<paddle::platform::bfloat16>();
template Tensor Tensor::copy_to_cpu<paddle::platform::complex64>();
template Tensor Tensor::copy_to_cpu<paddle::platform::complex128>();
template Tensor Tensor::copy_to_cpu<float>();
template Tensor Tensor::copy_to_cpu<double>();
template Tensor Tensor::copy_to_cpu<int64_t>();
template Tensor Tensor::copy_to_cpu<int32_t>();
template Tensor Tensor::copy_to_cpu<uint8_t>();
template Tensor Tensor::copy_to_cpu<int8_t>();
template Tensor Tensor::copy_to_cpu<int16_t>();

template float *Tensor::data<float>() const;
template double *Tensor::data<double>() const;
template int64_t *Tensor::data<int64_t>() const;
template int32_t *Tensor::data<int32_t>() const;
template uint8_t *Tensor::data<uint8_t>() const;
template int8_t *Tensor::data<int8_t>() const;
template paddle::platform::float16 *Tensor::data<paddle::platform::float16>()
    const;
template paddle::platform::bfloat16 *Tensor::data<paddle::platform::bfloat16>()
    const;
template paddle::platform::complex128 *
Tensor::data<paddle::platform::complex128>() const;
template paddle::platform::complex64 *
Tensor::data<paddle::platform::complex64>() const;
template int16_t *Tensor::data<int16_t>() const;

template float *Tensor::mutable_data<float>();
template double *Tensor::mutable_data<double>();
template int64_t *Tensor::mutable_data<int64_t>();
template int32_t *Tensor::mutable_data<int32_t>();
template uint8_t *Tensor::mutable_data<uint8_t>();
template int8_t *Tensor::mutable_data<int8_t>();
template paddle::platform::float16 *
Tensor::mutable_data<paddle::platform::float16>();
template paddle::platform::bfloat16 *
Tensor::mutable_data<paddle::platform::bfloat16>();
template paddle::platform::complex128 *
Tensor::mutable_data<paddle::platform::complex128>();
template paddle::platform::complex64 *
Tensor::mutable_data<paddle::platform::complex64>();
template int16_t *Tensor::mutable_data<int16_t>();

template float *Tensor::mutable_data<float>(const PlaceType &place);
template double *Tensor::mutable_data<double>(const PlaceType &place);
template int64_t *Tensor::mutable_data<int64_t>(const PlaceType &place);
template int32_t *Tensor::mutable_data<int32_t>(const PlaceType &place);
template uint8_t *Tensor::mutable_data<uint8_t>(const PlaceType &place);
template int8_t *Tensor::mutable_data<int8_t>(const PlaceType &place);
template paddle::platform::float16 *
Tensor::mutable_data<paddle::platform::float16>(const PlaceType &place);
template paddle::platform::bfloat16 *
Tensor::mutable_data<paddle::platform::bfloat16>(const PlaceType &place);
template paddle::platform::complex128 *
Tensor::mutable_data<paddle::platform::complex128>(const PlaceType &place);
template paddle::platform::complex64 *
Tensor::mutable_data<paddle::platform::complex64>(const PlaceType &place);
template int16_t *Tensor::mutable_data<int16_t>(const PlaceType &place);

std::vector<int> Tensor::shape() const {
  GET_CASTED_TENSOR
  return framework::vectorize<int>(tensor->dims());
}

const PlaceType &Tensor::place() const {
  GET_CASTED_TENSOR;
  if (platform::is_cpu_place(tensor->place())) {
    place_ = PlaceType::kCPU;
  } else if (platform::is_gpu_place(tensor->place())) {
    place_ = PlaceType::kGPU;
  } else {
    PADDLE_THROW(
        "Current Tensor hold unsupported Place Type, Please Init it"
        "using Tensor::mutable_data<T>(PaddlePlace) which T is"
        "either Place::kCPU or Place::kGPU");
  }
  return place_;
}

int64_t Tensor::size() const {
  GET_CASTED_TENSOR;
  return tensor->numel();
}

namespace framework {

void CustomTensorUtils::ShareDataTo(const paddle::Tensor &src, void *dst) {
  static_cast<framework::LoDTensor *>(dst)->ShareDataWith(
      *static_cast<framework::LoDTensor *>(src.tensor_.get()));
}

void CustomTensorUtils::ShareDataFrom(void *src, const paddle::Tensor &dst) {
  if (!dst.tensor_) {
    dst.tensor_ = std::make_shared<framework::LoDTensor>();
  }
  auto *tensor = static_cast<framework::LoDTensor *>(dst.tensor_.get());
  tensor->ShareDataWith(*static_cast<framework::LoDTensor *>(src));
}

}  // namespace framework
}  // namespace paddle
