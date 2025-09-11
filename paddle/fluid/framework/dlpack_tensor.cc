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
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/utils/visit_place.h"

namespace paddle {
namespace framework {

std::unordered_map<void *, std::function<void(phi::Allocation *)>>
    ptr_to_deleter;
std::mutex ptr_to_deleter_mutex;  // use mutex to keep thread safe

void DeleterBridge(phi::Allocation *alloc) {
  std::lock_guard<std::mutex> lock(ptr_to_deleter_mutex);
  auto it = ptr_to_deleter.find(static_cast<void *>(alloc->ptr()));
  if (it != ptr_to_deleter.end()) {
    it->second(alloc);         // call the deleter
    ptr_to_deleter.erase(it);  // remove the entry from the map safely
  }
}

namespace internal {
template <typename T>
::DLDataType GetDLDataTypeCode() {
  ::DLDataType dtype;
  if (std::is_same<T, phi::dtype::complex<float>>::value ||
      std::is_same<T, phi::dtype::complex<double>>::value) {
    dtype.code = kDLComplex;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    dtype.code = kDLBfloat;
  } else if (std::is_same<T, phi::dtype::float16>::value ||
             std::is_floating_point<T>::value) {
    dtype.code = kDLFloat;
  } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
    dtype.code = kDLFloat8_e4m3fn;
  } else if (std::is_same<T, phi::dtype::float8_e5m2>::value) {
    dtype.code = kDLFloat8_e5m2;
  } else if (std::is_same<T, bool>::value) {
    // Since std::is_unsigned<bool>::value is True,
    // it is necessary to evaluate bool before std::is_unsigned.
    dtype.code = kDLBool;
  } else if (std::is_unsigned<T>::value) {
    dtype.code = kDLUInt;
  } else if (std::is_integral<T>::value) {
    dtype.code = kDLInt;
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Unsupported data type (%s), only supports float16, float, unsigned "
        "int and int.",
        common::demangle(typeid(T).name())));
  }
  dtype.bits = 8 * sizeof(T);
  dtype.lanes = 1;
  return dtype;
}

// get tensor data point by DLDataType
void *GetDstPtrByDLDataType(DLDataType type,
                            phi::DenseTensor *dst,
                            const phi::Place &dst_place) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(
      type.lanes,
      1,
      common::errors::Unimplemented("Vector type is not supported currently."));

  switch (type.bits) {
    case 8:
      if (type.code == kDLBool)
        return static_cast<void *>(dst->mutable_data<bool>(dst_place));
      if (type.code == kDLInt)
        return static_cast<void *>(dst->mutable_data<int8_t>(dst_place));
      if (type.code == kDLUInt)
        return static_cast<void *>(dst->mutable_data<uint8_t>(dst_place));
      if (type.code == kDLFloat8_e4m3fn)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::float8_e4m3fn>(dst_place));
      if (type.code == kDLFloat8_e5m2)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::float8_e5m2>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt)
        return static_cast<void *>(dst->mutable_data<int16_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::float16>(dst_place));
      if (type.code == kDLBfloat)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::bfloat16>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt)
        return static_cast<void *>(dst->mutable_data<int32_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void *>(dst->mutable_data<float>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt)
        return static_cast<void *>(dst->mutable_data<int64_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void *>(dst->mutable_data<double>(dst_place));
      if (type.code == kDLComplex)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::complex<float>>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex)
        return static_cast<void *>(
            dst->mutable_data<phi::dtype::complex<double>>(dst_place));
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported DLDataType.bits %d.", type.bits));
  }
}

// get Tensor data dtype from given DLDataType
phi::DataType GetDstPtrByDLDataType(DLDataType type) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(
      type.lanes,
      1,
      common::errors::Unimplemented("Vector type is not supported currently."));

  switch (type.bits) {
    case 8:
      if (type.code == kDLBool) return phi::DataType::BOOL;
      if (type.code == kDLInt) return phi::DataType::INT8;
      if (type.code == kDLUInt) return phi::DataType::UINT8;
      if (type.code == kDLFloat8_e4m3fn) return phi::DataType::FLOAT8_E4M3FN;
      if (type.code == kDLFloat8_e5m2) return phi::DataType::FLOAT8_E5M2;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt) return phi::DataType::INT16;
      if (type.code == kDLFloat) return phi::DataType::FLOAT16;
      if (type.code == kDLBfloat) return phi::DataType::BFLOAT16;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt) return phi::DataType::INT32;
      if (type.code == kDLFloat) return phi::DataType::FLOAT32;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt) return phi::DataType::INT64;
      if (type.code == kDLFloat) return phi::DataType::FLOAT64;
      if (type.code == kDLComplex) return phi::DataType::COMPLEX64;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex) return phi::DataType::COMPLEX128;
      PADDLE_THROW(common::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported DLDataType.bits %d.", type.bits));
  }
}

template <typename T>
phi::DenseTensor from_blob(void *data,
                           T *src,
                           const phi::DDim &shape,
                           const phi::DDim &strides,
                           phi::DataType dtype,
                           const phi::Place &place,
                           const Deleter &deleter) {
  auto meta = phi::DenseTensorMeta(dtype, shape, strides);

  phi::Allocation::DeleterFnPtr f = nullptr;
  if (deleter) {
    auto g = [deleter, src](phi::Allocation *p) {
      if (src->manager_ctx) {
        deleter(src);
      }
    };

    {
      std::lock_guard<std::mutex> lock(ptr_to_deleter_mutex);
      ptr_to_deleter[data] = g;
    }

    f = DeleterBridge;
  }

  // Calculate the number of elements of underlying storage
  size_t size = 1;
  for (auto i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      size = 0;
      break;
    }
    size += strides[i] * (shape[i] - 1);
  }

  auto alloc =
      std::make_shared<phi::Allocation>(data, size * SizeOf(dtype), f, place);
  return phi::DenseTensor(alloc, meta);
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
  PADDLE_ENFORCE_NE(it,
                    type_to_dtype_map_end_it,
                    common::errors::InvalidArgument(
                        "Unsupported data type (%s).", DataTypeToString(type)));
  return it->second;
#undef REG_DL_DATA_TYPE
}

struct DLDeviceVisitor {
  using argument_type = const phi::Place &;
  using result_type = ::DLDevice;
  inline ::DLDevice operator()(const phi::CPUPlace &place) const {
    ::DLDevice device;
    device.device_type = kDLCPU;
    device.device_id = 0;
    return device;
  }

  inline ::DLDevice operator()(const phi::IPUPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::IPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::XPUPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::XPUPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::XPUPinnedPlace &place) const {
#if defined(PADDLE_WITH_XPU)
    ::DLDevice device;
    device.device_type = kDLCUDAHost;
    device.device_id = 0;
    return device;
#else
    PADDLE_THROW(common::errors::Unavailable(
        "phi::XPUPinnedPlace is not supported in CPU only version."));
#endif
  }

  inline ::DLDevice operator()(const phi::CustomPlace &place) const {
    PADDLE_THROW(
        common::errors::Unimplemented("phi::CustomPlace is not supported"));
  }

  inline ::DLDevice operator()(const phi::GPUPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLCUDA;
    device.device_id = place.device;  // NOLINT
    return device;
#else
    PADDLE_THROW(common::errors::Unavailable(
        "phi::GPUPlace is not supported in CPU only version."));
#endif
  }

  inline ::DLDevice operator()(const phi::GPUPinnedPlace &place) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ::DLDevice device;
    device.device_type = kDLCUDAHost;
    device.device_id = 0;
    return device;
#else
    PADDLE_THROW(common::errors::Unavailable(
        "phi::GPUPinnedPlace is not supported in CPU only version."));
#endif
  }
};
}  // namespace internal

template <typename T>
struct PaddleDLMTensor {
  phi::DenseTensor handle;
  T tensor;
};

template <typename T>
static void deleter(T *self) {
  if (self && self->manager_ctx) {
    delete[] self->dl_tensor
        .shape;  // delete shape allocated in toDLPack manually
    delete[] self->dl_tensor
        .strides;  // delete strides allocated in toDLPack manually
    delete static_cast<PaddleDLMTensor<T> *>(self->manager_ctx);
  }
}

template <class T>
void fillVersion(T *tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(DLManagedTensorVersioned *tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

template <typename T>
T *toDLPackImpl(const phi::DenseTensor &src) {
  PaddleDLMTensor<T> *pdDLMTensor(new PaddleDLMTensor<T>);
  pdDLMTensor->handle = const_cast<phi::DenseTensor &>(src);
  pdDLMTensor->tensor.manager_ctx = pdDLMTensor;
  pdDLMTensor->tensor.deleter = &deleter<T>;

  // init ndim
  using DimType = decltype(pdDLMTensor->tensor.dl_tensor.ndim);  // int32_t
  auto _shape = src.dims();
  pdDLMTensor->tensor.dl_tensor.ndim = static_cast<DimType>(_shape.size());
  DimType ndim = pdDLMTensor->tensor.dl_tensor.ndim;

  // init shape
  int64_t *shape = new int64_t[ndim];
  for (DimType i = 0; i < ndim; ++i) {
    shape[i] = _shape[i];
  }
  pdDLMTensor->tensor.dl_tensor.shape = shape;

  // init strides
  auto _strides = src.strides();
  int64_t *strides = new int64_t[ndim];
  for (int i = 0; i < src.dims().size(); i++) {
    strides[i] = _strides[i];
    if (shape[i] < 2) {
      strides[i] = 1;
    }
  }
  pdDLMTensor->tensor.dl_tensor.strides = strides;

  pdDLMTensor->tensor.dl_tensor.data = const_cast<void *>(src.data());
  auto place = src.place();
  pdDLMTensor->tensor.dl_tensor.device =
      phi::VisitPlace(place, internal::DLDeviceVisitor());
  pdDLMTensor->tensor.dl_tensor.dtype = internal::GetDLDataTypeFromTypeIndex(
      framework::TransToProtoVarType(src.dtype()));
  pdDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&(pdDLMTensor->tensor));
  return &(pdDLMTensor->tensor);
}

DLManagedTensor *toDLPack(const phi::DenseTensor &src) {
  return toDLPackImpl<DLManagedTensor>(src);
}

DLManagedTensorVersioned *toDLPackVersioned(const phi::DenseTensor &src) {
  return toDLPackImpl<DLManagedTensorVersioned>(src);
}

template <typename T>
phi::DenseTensor fromDLPackImpl(T *src, Deleter deleter) {
  std::vector<int64_t> shape_vec;
  std::copy(src->dl_tensor.shape,
            src->dl_tensor.shape + src->dl_tensor.ndim,
            std::back_inserter(shape_vec));

  phi::Place place;
  if (src->dl_tensor.device.device_type == kDLCPU) {
    place = phi::CPUPlace();
  } else if (src->dl_tensor.device.device_type == kDLCUDA) {
    place = phi::GPUPlace(src->dl_tensor.device.device_id);
  } else if (src->dl_tensor.device.device_type == kDLCUDAHost) {
    place = phi::GPUPinnedPlace();
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Given Place is not supported"));
  }

  ::DLDataType type = src->dl_tensor.dtype;
  auto dtype = internal::GetDstPtrByDLDataType(type);
  if (!src->dl_tensor.strides) {
    return internal::from_blob(
        src->dl_tensor.data,
        src,
        common::make_ddim(shape_vec),
        phi::DenseTensorMeta::calc_strides(common::make_ddim(shape_vec)),
        dtype,
        place,
        std::move(deleter));
  } else {
    std::vector<int64_t> strides_vec;
    std::copy(src->dl_tensor.strides,
              src->dl_tensor.strides + src->dl_tensor.ndim,
              std::back_inserter(strides_vec));
    return internal::from_blob(src->dl_tensor.data,
                               src,
                               common::make_ddim(shape_vec),
                               common::make_ddim(strides_vec),
                               dtype,
                               place,
                               deleter);
  }
}

phi::DenseTensor fromDLPack(DLManagedTensor *src, Deleter deleter) {
  return fromDLPackImpl<DLManagedTensor>(src, std::move(deleter));
}

phi::DenseTensor fromDLPackVersioned(DLManagedTensorVersioned *src,
                                     Deleter deleter) {
  return fromDLPackImpl<DLManagedTensorVersioned>(src, std::move(deleter));
}

}  // namespace framework
}  // namespace paddle
