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
#include "paddle/fluid/framework/dlpack_tensor.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace framework {

namespace internal {
template <typename T>
static ::DLDataType GetDLDataTypeCode() {
  ::DLDataType dtype;
  if (std::is_same<T, platform::complex<float>>::value ||
      std::is_same<T, platform::complex<double>>::value) {
    dtype.code = kDLComplex;
  } else if (std::is_same<T, platform::bfloat16>::value) {
    dtype.code = kDLBfloat;
  } else if (std::is_same<T, platform::float16>::value ||
             std::is_floating_point<T>::value) {
    dtype.code = kDLFloat;
  } else if (std::is_unsigned<T>::value) {
    dtype.code = kDLUInt;
  } else if (std::is_integral<T>::value) {
    dtype.code = kDLInt;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported data type (%s), only supports float16, float, unsigned "
        "int and int.",
        platform::demangle(typeid(T).name())));
  }
  dtype.bits = 8 * sizeof(T);
  dtype.lanes = 1;
  return dtype;
}

static std::unordered_map<int, ::DLDataType> CreateDLDataTypeMap() {
  static std::unordered_map<int, ::DLDataType> result;

#define REG_DL_DATA_TYPE(cpp_type, proto_type) \
  result[static_cast<int>(proto_type)] = GetDLDataTypeCode<cpp_type>()

  _ForEachDataType_(REG_DL_DATA_TYPE);
#undef REG_DL_DATA_TYPE
  return result;
}

static DLDataType GetDLDataTypeFromTypeIndex(proto::VarType::Type type) {
  static auto type_to_dtype_map = CreateDLDataTypeMap();
  static auto type_to_dtype_map_end_it = type_to_dtype_map.end();
  auto it = type_to_dtype_map.find(static_cast<int>(type));
  PADDLE_ENFORCE_NE(it, type_to_dtype_map_end_it,
                    platform::errors::InvalidArgument(
                        "Unsupported data type (%s).", DataTypeToString(type)));
  return it->second;
#undef REG_DL_DATA_TYPE
}

struct DLDeviceVisitor : public boost::static_visitor<::DLDevice> {
  inline ::DLDevice operator()(const platform::CPUPlace &place) const {
    ::DLDevice device;
    device.device_type = kDLCPU;
    device.device_id = 0;
    return device;
  }

  inline ::DLDevice operator()(const platform::IPUPlace &place) const {
    PADDLE_THROW(
        platform::errors::Unimplemented("platform::IPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const platform::XPUPlace &place) const {
    PADDLE_THROW(
        platform::errors::Unimplemented("platform::XPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const platform::NPUPlace &place) const {
    PADDLE_THROW(
        platform::errors::Unimplemented("platform::NPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const platform::NPUPinnedPlace &place) const {
    PADDLE_THROW(platform::errors::Unimplemented(
        "platform::NPUPinnedPlace is not supported"));
  }

  inline ::DLDevice operator()(const platform::MLUPlace &place) const {
    PADDLE_THROW(
        platform::errors::Unimplemented("platform::MLUPlace is not supported"));
  }

  inline ::DLDevice operator()(const platform::CUDAPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLGPU;
    device.device_id = place.device;
    return device;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "platform::CUDAPlace is not supported in CPU only version."));
#endif
  }

  inline ::DLDevice operator()(const platform::CUDAPinnedPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLCPUPinned;
    device.device_id = 0;
    return device;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "platform::CUDAPinnedPlace is not supported in CPU only version."));
#endif
  }
};
}  // namespace internal

DLPackTensor::DLPackTensor(const Tensor &tensor, LaneType lanes) {
  // init data, data buffer
  t_.data = const_cast<void *>(tensor.data());

  // init device, DLDevice type with device_type and device_id
  auto place = tensor.place();
  t_.device = paddle::platform::VisitPlace(place, internal::DLDeviceVisitor());

  // init dtype
  t_.dtype = internal::GetDLDataTypeFromTypeIndex(
      framework::TransToProtoVarType(tensor.dtype()));
  t_.dtype.lanes = lanes;

  // init ndim, tensor rank
  auto &dims = tensor.dims();
  using DimType = decltype(t_.ndim);  // int
  t_.ndim = static_cast<DimType>(dims.size());

  // init shape, tensor dims
  t_.shape = shape_;
  for (DimType i = 0; i < t_.ndim; ++i) {
    t_.shape[i] = dims[i];
  }

  // init strides, nullptr means the tensor is compact
  t_.strides = nullptr;

  // init byte_offset
  t_.byte_offset = 0;
}

::DLManagedTensor *DLPackTensor::ToDLManagedTensor() {
  // init shape
  auto shape = new int64_t[t_.ndim];
  using DimType = decltype(t_.ndim);  // int
  for (DimType i = 0; i < t_.ndim; ++i) {
    shape[i] = t_.shape[i];
  }
  t_.shape = shape;

  // init strides
  auto strides = new int64_t[t_.ndim];
  for (DimType i = 0; i < t_.ndim; ++i) {
    strides[i] = 1;
  }
  for (DimType i = t_.ndim - 2; i >= 0; --i) {
    strides[i] = t_.shape[i + 1] * strides[i + 1];
  }
  t_.strides = strides;

  auto tensor = new DLManagedTensor;
  tensor->dl_tensor = t_;

  tensor->deleter = [](DLManagedTensor *arg) {
    delete[] arg->dl_tensor.shape;
    delete[] arg->dl_tensor.strides;
    delete arg;
  };

  tensor->manager_ctx = nullptr;

  return tensor;
}

}  // namespace framework
}  // namespace paddle
