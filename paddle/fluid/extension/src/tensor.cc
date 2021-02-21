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
#include "paddle/fluid/platform/transform.h"

namespace paddle {

template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

template <typename InType>
struct CastDataType {
  CastDataType(const framework::Tensor &in, framework::Tensor *out,
               const platform::DeviceContext *ctx)
      : in_(in), out_(out), ctx_(ctx) {}
  const framework::Tensor in_;
  framework::Tensor *out_;
  const platform::DeviceContext *ctx_;

  template <typename OutType>
  void apply() {
    auto *in_begin = in_.data<InType>();
    auto *in_end = in_begin + in_.numel();
    auto *out_begin = out_->mutable_data<OutType>(in_.place());

    if (platform::is_cpu_place(in_.place())) {
      platform::Transform<platform::CPUDeviceContext> trans;
      auto *context = static_cast<const platform::CPUDeviceContext *>(ctx_);
      trans(*context, in_begin, in_end, out_begin,
            CastDataTypeFunctor<InType, OutType>());
#ifdef __NVCC__
    } else if (platform::is_gpu_place(in_.place())) {
      platform::Transform<platform::CUDADeviceContext> trans;
      auto *context = static_cast<const platform::CUDADeviceContext *>(ctx_);
      trans(*context, in_begin, in_end, out_begin,
            CastDataTypeFunctor<InType, OutType>());
      context->Wait();
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Place type is not supported when casting data type."));
    }
  }
};
template <typename T>
void GpuCopy(T *src, T *dst, PlaceType src_plc, PlaceType dst_plc,
             int64_t ele_size) {
#ifdef PADDLE_WITH_CUDA
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  int device_num = paddle::platform::GetCurrentDeviceId();
  platform::CUDAPlace gpu_place(device_num);
  auto *dev_ctx =
      static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
  if ((src_plc == PlaceType::kGPU) && (dst_plc == PlaceType::kCPU)) {
    memory::Copy(platform::CPUPlace(), static_cast<void *>(dst), gpu_place, src,
                 ele_size, dev_ctx->stream());
  } else if ((src_plc == PlaceType::kGPU) && (dst_plc == PlaceType::kGPU)) {
    memory::Copy(gpu_place, static_cast<void *>(dst), gpu_place, src, ele_size,
                 dev_ctx->stream());
  } else if ((src_plc == PlaceType::kCPU) && (dst_plc == PlaceType::kGPU)) {
    memory::Copy(gpu_place, static_cast<void *>(dst), platform::CPUPlace(), src,
                 ele_size, dev_ctx->stream());
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Only GPU related Copy can reach this func."));
  }
  cudaStreamSynchronize(dev_ctx->stream());
#endif
}

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
#ifdef PADDLE_WITH_CUDA
    case static_cast<int>(PlaceType::kGPU): {
      int device_num = platform::GetCurrentDeviceId();
      return tensor->mutable_data<T>(platform::CUDAPlace(device_num));
    }
#endif
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Custom operator unsupported place id(%d)",
          static_cast<int>(place_)));
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
  } else if (type == framework::proto::VarType::BOOL) {
    return DataType::BOOL;
  }
  return DataType::FLOAT32;
}

template <typename T>
Tensor Tensor::copy_to(const PlaceType &target_place) const {
  GET_CASTED_TENSOR;
  PADDLE_ENFORCE_GE(tensor->numel(), 0,
                    platform::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);
  auto *p_src_data = tensor->data<T>();
  auto src_place = place();
  Tensor target = Tensor(target_place);
  target.reshape(shape());
  auto *p_target_data = target.template mutable_data<T>();

  if ((src_place == PlaceType::kCPU) && (target_place == PlaceType::kCPU)) {
    std::memcpy(static_cast<void *>(p_target_data), p_src_data, ele_size);
  } else if ((src_place == PlaceType::kGPU) &&
             (target_place == PlaceType::kCPU)) {
    GpuCopy<T>(p_src_data, p_target_data, src_place, target_place, ele_size);
  } else if ((src_place == PlaceType::kCPU) &&
             (target_place == PlaceType::kGPU)) {
    GpuCopy<T>(p_src_data, p_target_data, src_place, target_place, ele_size);
  } else if ((src_place == PlaceType::kGPU) &&
             (target_place == PlaceType::kGPU)) {
    GpuCopy<T>(p_src_data, p_target_data, src_place, target_place, ele_size);
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Not supported place transform of place: %d to place: %d",
        static_cast<int>(src_place), static_cast<int>(target_place)));
  }
  return target;
}

template Tensor Tensor::copy_to<paddle::platform::float16>(
    const PlaceType &target_place) const;
template Tensor Tensor::copy_to<paddle::platform::bfloat16>(
    const PlaceType &target_place) const;
template Tensor Tensor::copy_to<paddle::platform::complex64>(
    const PlaceType &target_place) const;
template Tensor Tensor::copy_to<paddle::platform::complex128>(
    const PlaceType &target_place) const;
template Tensor Tensor::copy_to<float>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<double>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<int64_t>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<int32_t>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<uint8_t>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<int8_t>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<int16_t>(const PlaceType &target_place) const;
template Tensor Tensor::copy_to<bool>(const PlaceType &target_place) const;

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
template bool *Tensor::data<bool>() const;

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
template bool *Tensor::mutable_data<bool>();

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
template bool *Tensor::mutable_data<bool>(const PlaceType &place);

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
    PADDLE_THROW(platform::errors::Unimplemented(
        "Current Tensor hold unsupported Place Type, Please Init it"
        "using Tensor::mutable_data<T>(PaddlePlace) which T is"
        "either Place::kCPU or Place::kGPU"));
  }
  return place_;
}

Tensor Tensor::cast(const DataType &target_type) const {
  GET_CASTED_TENSOR;
  Tensor rlt = Tensor(place());
  rlt.reshape(this->shape());
  auto rlt_tensor_ = static_cast<framework::LoDTensor *>(rlt.tensor_.get());
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto ctx = pool.Get(tensor->place());
  auto src_type = tensor->type();
  auto dst_type =
      framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(target_type);
  switch (src_type) {
    case framework::proto::VarType::FP16:
      framework::VisitDataType(
          dst_type, CastDataType<platform::float16>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::BF16:
      framework::VisitDataType(dst_type, CastDataType<platform::bfloat16>(
                                             *tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::FP32:
      framework::VisitDataType(dst_type,
                               CastDataType<float>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::FP64:
      framework::VisitDataType(dst_type,
                               CastDataType<double>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::INT32:
      framework::VisitDataType(dst_type,
                               CastDataType<int>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::INT64:
      framework::VisitDataType(
          dst_type, CastDataType<int64_t>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::BOOL:
      framework::VisitDataType(dst_type,
                               CastDataType<bool>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::INT16:
      framework::VisitDataType(
          dst_type, CastDataType<int16_t>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::UINT8:
      framework::VisitDataType(
          dst_type, CastDataType<uint8_t>(*tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::COMPLEX64:
      framework::VisitDataType(dst_type, CastDataType<platform::complex64>(
                                             *tensor, rlt_tensor_, ctx));
      break;
    case framework::proto::VarType::COMPLEX128:
      framework::VisitDataType(dst_type, CastDataType<platform::complex128>(
                                             *tensor, rlt_tensor_, ctx));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          framework::DataTypeToString(src_type)));
  }
  return rlt;
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

void CustomTensorUtils::ShareDataFrom(const void *src,
                                      const paddle::Tensor &dst) {
  if (!dst.tensor_) {
    dst.tensor_ = std::make_shared<framework::LoDTensor>();
  }
  auto *tensor = static_cast<framework::LoDTensor *>(dst.tensor_.get());
  tensor->ShareDataWith(*static_cast<const framework::LoDTensor *>(src));
}

}  // namespace framework
}  // namespace paddle
