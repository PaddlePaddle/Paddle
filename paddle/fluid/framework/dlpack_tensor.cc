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
#include "paddle/fluid/framework/data_type.h"
namespace paddle {
namespace framework {

namespace internal {
template <typename T>
static ::DLDataType GetDLDataTypeCode() {
  ::DLDataType dtype;
  if (std::is_same<T, platform::float16>::value ||
      std::is_floating_point<T>::value) {
    dtype.code = kDLFloat;
  } else if (std::is_unsigned<T>::value) {
    dtype.code = kDLUInt;
  } else if (std::is_integral<T>::value) {
    dtype.code = kDLInt;
  } else {
    PADDLE_THROW("Unsupported data type %s", typeid(T).name());
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
  PADDLE_ENFORCE(it != type_to_dtype_map_end_it, "Unsupported data type %d",
                 type);
  return it->second;
#undef REG_DL_DATA_TYPE
}

struct DLContextVisitor : public boost::static_visitor<::DLContext> {
  inline ::DLContext operator()(const platform::CPUPlace &place) const {
    ::DLContext ctx;
    ctx.device_type = kDLCPU;
    ctx.device_id = 0;
    return ctx;
  }

  inline ::DLContext operator()(const platform::CUDAPlace &place) const {
#ifdef PADDLE_WITH_CUDA
    ::DLContext ctx;
    ctx.device_type = kDLGPU;
    ctx.device_id = place.device;
    return ctx;
#else
    PADDLE_THROW("platform::CUDAPlace is not supported in CPU only version");
#endif
  }

  inline ::DLContext operator()(const platform::CUDAPinnedPlace &place) const {
#ifdef PADDLE_WITH_CUDA
    ::DLContext ctx;
    ctx.device_type = kDLCPUPinned;
    ctx.device_id = 0;
    return ctx;
#else
    PADDLE_THROW(
        "platform::CUDAPinnedPlace is not supported in CPU only version");
#endif
  }
};
}  // namespace internal

DLPackTensor::DLPackTensor(const Tensor &tensor, LaneType lanes) {
  // init data, data buffer
  t_.data = const_cast<void *>(tensor.data<void>());

  // init ctx, DLContext type with device_type and device_id
  auto place = tensor.place();
  t_.ctx = boost::apply_visitor(internal::DLContextVisitor(), place);

  // init dtype
  t_.dtype = internal::GetDLDataTypeFromTypeIndex(tensor.type());
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

}  // namespace framework
}  // namespace paddle
