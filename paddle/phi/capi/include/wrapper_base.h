// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#if !defined(_WIN32)

#include <memory>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/common/exception.h"
#include "paddle/phi/capi/include/c_device_context.h"
#include "paddle/phi/capi/include/c_infer_meta_context.h"
#include "paddle/phi/capi/include/c_int_array.h"
#include "paddle/phi/capi/include/c_kernel_context.h"
#include "paddle/phi/capi/include/c_kernel_factory.h"
#include "paddle/phi/capi/include/c_kernel_registry.h"
#include "paddle/phi/capi/include/c_meta_tensor.h"
#include "paddle/phi/capi/include/c_place.h"
#include "paddle/phi/capi/include/c_scalar.h"
#include "paddle/phi/capi/include/c_tensor.h"
#include "paddle/phi/capi/include/data_type.h"
#include "paddle/utils/optional.h"

#define PD_CHECK_STATUS(status) PD_CHECK(status == C_SUCCESS)

namespace phi {

namespace capi {

using LoD = std::vector<std::vector<size_t>>;

template <typename T>
static inline PD_List PDListFromVector(std::vector<T>* vec) {
  PD_List list;
  list.data = reinterpret_cast<void*>(vec->data());
  list.size = vec->size();
  return list;
}

template <typename T>
static inline std::vector<T> PDListToVector(PD_List list) {
  return std::vector<T>(static_cast<T*>(list.data),
                        static_cast<T*>(list.data) + list.size);
}

inline std::vector<int64_t> PD_TensorGetDims(PD_Tensor* tensor,
                                             PD_Status* status) {
  int64_t ndims = PD_TensorGetNumDims(tensor, status);
  if (ndims > 0) {
    std::vector<int64_t> shape(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      shape[i] = PD_TensorGetDim(tensor, i, status);
    }
    return shape;
  }
  return std::vector<int64_t>();
}

inline std::vector<int64_t> PD_TensorGetStrides(PD_Tensor* tensor,
                                                PD_Status* status) {
  int64_t nstrides = PD_TensorGetNumStrides(tensor, status);
  if (nstrides > 0) {
    std::vector<int64_t> shape(nstrides);
    for (int64_t i = 0; i < nstrides; ++i) {
      shape[i] = PD_TensorGetStride(tensor, i, status);
    }
    return shape;
  }
  return std::vector<int64_t>();
}

inline std::vector<int64_t> PD_MetaTensorGetDims(PD_MetaTensor* tensor,
                                                 PD_Status* status) {
  int64_t ndims = PD_MetaTensorGetNumDims(tensor, status);
  if (ndims > 0) {
    std::vector<int64_t> shape(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      shape[i] = PD_MetaTensorGetDim(tensor, i, status);
    }
    return shape;
  }
  return std::vector<int64_t>();
}

inline std::vector<int64_t> PD_MetaTensorGetStrides(PD_MetaTensor* tensor,
                                                    PD_Status* status) {
  int64_t nstrides = PD_MetaTensorGetNumStrides(tensor, status);
  if (nstrides > 0) {
    std::vector<int64_t> shape(nstrides);
    for (int64_t i = 0; i < nstrides; ++i) {
      shape[i] = PD_MetaTensorGetStride(tensor, i, status);
    }
    return shape;
  }
  return std::vector<int64_t>();
}

template <typename T>
class WrapperBase {
 public:
  explicit WrapperBase(T* ptr, bool own = false) : data_(ptr), own_(own) {}

  inline T* raw_data() const { return data_; }

  inline bool own_data() const { return own_; }

  inline void reset(const T* ptr) { data_ = ptr; }

 private:
  T* data_;
  bool own_;
};

class DenseTensor : public WrapperBase<PD_Tensor> {
 public:
  DenseTensor() : WrapperBase(PD_NewTensor(), true) {}

  explicit DenseTensor(PD_Tensor* tensor) : WrapperBase(tensor) {}

  ~DenseTensor() {
    if (own_data()) {
      PD_DeleteTensor(raw_data());
    }
  }

  bool valid() const {
    C_Status status;
    auto ret = PD_TensorIsValid(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return ret;
  }

  bool initialized() const {
    C_Status status;
    auto ret = PD_TensorIsInitialized(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return ret;
  }

  void* Holder() const {
    C_Status status;
    auto holder = PD_TensorGetHolder(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return holder;
  }

  size_t offset() const {
    C_Status status;
    auto offset = PD_TensorGetOffset(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return offset;
  }

  std::vector<int64_t> dims() const {
    C_Status status;
    auto dimension = PD_TensorGetDims(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return dimension;
  }

  std::vector<int64_t> strides() const {
    C_Status status;
    auto strides = PD_TensorGetStrides(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return strides;
  }

  PD_DataType dtype() const {
    C_Status status;
    auto data_type = PD_TensorGetPDDataType(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_type;
  }

  PD_DataLayout layout() const {
    C_Status status;
    auto data_layout = PD_TensorGetDataLayout(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_layout;
  }

  int64_t numel() const {
    C_Status status;
    auto element_count = PD_TensorGetElementCount(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return element_count;
  }

  int64_t memory_size() const {
    C_Status status;
    auto byte_size = PD_TensorGetByteSize(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return byte_size;
  }

  LoD lod() const {
    PD_List data, offset;
    C_Status status;
    PD_TensorGetLoD(raw_data(), &data, &offset, &status);
    PD_CHECK_STATUS(status);
    LoD lod_;
    auto ptr = static_cast<size_t*>(data.data);
    auto offset_ptr = static_cast<size_t*>(offset.data);
    for (size_t i = 0; i < offset.size - 1; ++i) {
      lod_.emplace_back(ptr + offset_ptr[i], ptr + offset_ptr[i + 1]);
    }
    delete[] ptr;
    delete[] offset_ptr;
    return lod_;
  }

  void ResetLoD(const LoD& lod) {
    std::vector<size_t> data, offset;
    offset.push_back(0);
    for (const auto& item : lod) {
      data.insert(data.cend(), item.cbegin(), item.cend());
      offset.push_back(item.size());
    }
    PD_List data_list, offset_list;
    data_list = PDListFromVector(&data);
    offset_list = PDListFromVector(&offset);

    C_Status status;
    PD_TensorResetLoD(raw_data(), data_list, offset_list, &status);
    PD_CHECK_STATUS(status);
  }

  void Resize(const std::vector<int64_t>& dims) {
    C_Status status;
    PD_TensorSetDims(raw_data(), dims.size(), dims.data(), &status);
    PD_CHECK_STATUS(status);
  }

  void set_offset(const int64_t& offset) {
    C_Status status;
    PD_TensorSetOffset(raw_data(), offset, &status);
    PD_CHECK_STATUS(status);
  }

  void set_strides(const std::vector<int64_t>& strides) {
    C_Status status;
    PD_TensorSetStrides(raw_data(), strides.size(), strides.data(), &status);
    PD_CHECK_STATUS(status);
  }

  void set_dtype(PD_DataType data_type) {
    C_Status status;
    PD_TensorSetDataType(raw_data(), data_type, &status);
    PD_CHECK_STATUS(status);
  }

  void set_layout(PD_DataLayout data_layout) {
    C_Status status;
    PD_TensorSetDataLayout(raw_data(), data_layout, &status);
    PD_CHECK_STATUS(status);
  }

  template <typename T>
  T* data() const {
    C_Status status;
    auto ptr = PD_TensorGetDataPointer(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return static_cast<T*>(ptr);
  }

  DenseTensor& ShareDataWith(const DenseTensor& src) {
    C_Status status;
    PD_TensorShareDataWith(raw_data(), src.raw_data(), &status);
    PD_CHECK_STATUS(status);
    return *this;
  }

  void share_lod(const DenseTensor& src) {
    C_Status status;
    PD_TensorShareLoDWith(raw_data(), src.raw_data(), &status);
    PD_CHECK_STATUS(status);
  }
};

class DeviceContext : public WrapperBase<PD_DeviceContext> {
 public:
  explicit DeviceContext(PD_DeviceContext* context)
      : WrapperBase<PD_DeviceContext>(context) {}

  void* stream() const {
    C_Status status;
    auto stream_ = PD_DeviceContextGetStream(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return stream_;
  }

  void* Alloc(DenseTensor* tensor,
              PD_DataType dtype,
              int64_t requested_size = 0) const {
    C_Status status;
    auto ptr = PD_DeviceContextAllocateTensor(
        raw_data(), tensor->raw_data(), requested_size, dtype, &status);
    PD_CHECK_STATUS(status);
    return static_cast<void*>(ptr);
  }

  template <typename T>
  T* Alloc(DenseTensor* tensor, int64_t requested_size = 0) const {
    C_Status status;
    auto ptr =
        PD_DeviceContextAllocateTensor(raw_data(),
                                       tensor->raw_data(),
                                       requested_size,
                                       phi::capi::CppTypeToPDType<T>::Type(),
                                       &status);
    PD_CHECK_STATUS(status);
    return static_cast<T*>(ptr);
  }

  void* HostAlloc(DenseTensor* tensor,
                  PD_DataType dtype,
                  int64_t requested_size = 0) const {
    C_Status status;
    auto ptr = PD_DeviceContextAllocateTensor(
        nullptr, tensor->raw_data(), requested_size, dtype, &status);
    PD_CHECK_STATUS(status);
    return static_cast<void*>(ptr);
  }

  template <typename T>
  T* HostAlloc(DenseTensor* tensor, int64_t requested_size = 0) const {
    C_Status status;
    auto ptr =
        PD_DeviceContextAllocateTensor(nullptr,
                                       tensor->raw_data(),
                                       requested_size,
                                       phi::capi::CppTypeToPDType<T>::Type(),
                                       &status);
    PD_CHECK_STATUS(status);
    return static_cast<T*>(ptr);
  }

  uint64_t seed() const {
    C_Status status;
    auto seed_val = PD_DeviceContextGetSeed(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return seed_val;
  }

  void seed(uint64_t seed_val) const {
    C_Status status;
    PD_DeviceContextSetSeed(raw_data(), seed_val, &status);
    PD_CHECK_STATUS(status);
  }

  uint64_t random() const {
    C_Status status;
    auto rand_val = PD_DeviceContextGetRandom(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return rand_val;
  }
};

class Scalar : public WrapperBase<PD_Scalar> {
 public:
  explicit Scalar(PD_Scalar* scalar) : WrapperBase<PD_Scalar>(scalar) {}

  PD_DataType dtype() const { return PD_ScalarGetDataType(raw_data()); }

  template <typename T>
  inline T to() const;
};

template <>
inline bool Scalar::to<bool>() const {
  return PD_ScalarGetBoolData(raw_data());
}

template <>
inline float Scalar::to<float>() const {
  return PD_ScalarGetFloat32Data(raw_data());
}

template <>
inline double Scalar::to<double>() const {
  return PD_ScalarGetFloat64Data(raw_data());
}

template <>
inline uint8_t Scalar::to<uint8_t>() const {
  return PD_ScalarGetUInt8Data(raw_data());
}

template <>
inline uint16_t Scalar::to<uint16_t>() const {
  return PD_ScalarGetUInt16Data(raw_data());
}

template <>
inline uint32_t Scalar::to<uint32_t>() const {
  return PD_ScalarGetUInt32Data(raw_data());
}

template <>
inline uint64_t Scalar::to<uint64_t>() const {
  return PD_ScalarGetUInt64Data(raw_data());
}

template <>
inline int8_t Scalar::to<int8_t>() const {
  return PD_ScalarGetInt8Data(raw_data());
}

template <>
inline int16_t Scalar::to<int16_t>() const {
  return PD_ScalarGetInt16Data(raw_data());
}

template <>
inline int32_t Scalar::to<int32_t>() const {
  return PD_ScalarGetInt32Data(raw_data());
}

template <>
inline int64_t Scalar::to<int64_t>() const {
  return PD_ScalarGetInt64Data(raw_data());
}

class IntArray : WrapperBase<PD_IntArray> {
 public:
  explicit IntArray(PD_IntArray* int_array)
      : WrapperBase<PD_IntArray>(int_array) {}

  size_t size() const { return PD_IntArrayGetElementCount(raw_data()); }

  std::vector<int64_t> GetData() const {
    auto list = PD_IntArrayGetDataPointer(raw_data());
    auto data = reinterpret_cast<int64_t*>(list.data);
    std::vector<int64_t> ret(data, data + list.size);
    return ret;
  }
};

class Place : WrapperBase<PD_Place> {
 public:
  explicit Place(PD_Place* place) : WrapperBase<PD_Place>(place) {}

  bool is_host() { return PD_PlaceIsHost(raw_data()); }

  int8_t GetDeviceID() { return PD_PlaceGetDeviceId(raw_data()); }
};

class TensorArgDef : WrapperBase<PD_TensorArgDef> {
 public:
  explicit TensorArgDef(PD_TensorArgDef* tensor_arg_def)
      : WrapperBase<PD_TensorArgDef>(tensor_arg_def) {}

  // TensorArgDef& SetBackend() {
  //   return *this;
  // }

  TensorArgDef& SetDataLayout(PD_DataLayout in_layout) {
    C_Status status;
    PD_TensorArgDefSetDataLayout(raw_data(), in_layout, &status);
    PD_CHECK_STATUS(status);
    return *this;
  }

  TensorArgDef& SetDataType(PD_DataType in_dtype) {
    C_Status status;
    PD_TensorArgDefSetDataType(raw_data(), in_dtype, &status);
    PD_CHECK_STATUS(status);
    return *this;
  }
};

class KernelArgsDef : WrapperBase<PD_KernelArgsDef> {
 public:
  explicit KernelArgsDef(PD_KernelArgsDef* kernel_args_def)
      : WrapperBase<PD_KernelArgsDef>(kernel_args_def) {}

  std::vector<TensorArgDef> input_defs() {
    C_Status status;
    auto list = PD_KernelArgsDefGetInputArgDefs(raw_data(), &status);
    PD_CHECK_STATUS(status);
    auto ptr = reinterpret_cast<PD_TensorArgDef**>(list.data);
    std::vector<TensorArgDef> ret;
    for (size_t i = 0; i < list.size; ++i) {
      ret.emplace_back(ptr[i]);
    }
    PD_DeletePointerList(list);
    return ret;
  }

  std::vector<TensorArgDef> output_defs() {
    C_Status status;
    auto list = PD_KernelArgsDefGetOutputArgDefs(raw_data(), &status);
    PD_CHECK_STATUS(status);
    auto ptr = reinterpret_cast<PD_TensorArgDef**>(list.data);
    std::vector<TensorArgDef> ret;
    for (size_t i = 0; i < list.size; ++i) {
      ret.emplace_back(ptr[i]);
    }
    PD_DeletePointerList(list);
    return ret;
  }
};

class KernelKey : WrapperBase<PD_KernelKey> {
 public:
  explicit KernelKey(PD_KernelKey* kernel_key)
      : WrapperBase<PD_KernelKey>(kernel_key) {}

  PD_DataLayout layout() const {
    PD_Status status;
    auto layout_ = PD_KernelKeyGetLayout(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return layout_;
  }

  PD_DataType dtype() const {
    PD_Status status;
    auto dtype_ = PD_KernelKeyGetDataType(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return dtype_;
  }
};

class Kernel : WrapperBase<PD_Kernel> {
 public:
  explicit Kernel(PD_Kernel* kernel) : WrapperBase<PD_Kernel>(kernel) {}

  KernelArgsDef args_def() const {
    C_Status status;
    auto ptr = PD_KernelGetArgsDef(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return KernelArgsDef(ptr);
  }

  TensorArgDef InputAt(size_t idx) { return args_def().input_defs()[idx]; }

  TensorArgDef OutputAt(size_t idx) { return args_def().input_defs()[idx]; }
};

class MetaTensor : WrapperBase<PD_MetaTensor> {
 public:
  explicit MetaTensor(PD_MetaTensor* meta_tensor)
      : WrapperBase<PD_MetaTensor>(meta_tensor) {}

  std::vector<int64_t> dims() const {
    C_Status status;
    auto dimension = PD_MetaTensorGetDims(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return dimension;
  }

  std::vector<int64_t> strides() const {
    C_Status status;
    auto strides = PD_MetaTensorGetStrides(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return strides;
  }

  PD_DataType dtype() const {
    C_Status status;
    auto data_type = PD_MetaTensorGetPDDataType(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_type;
  }

  PD_DataLayout layout() const {
    C_Status status;
    auto data_layout = PD_MetaTensorGetDataLayout(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return data_layout;
  }

  int64_t numel() const {
    C_Status status;
    auto element_count = PD_MetaTensorGetElementCount(raw_data(), &status);
    PD_CHECK_STATUS(status);
    return element_count;
  }

  void set_dims(const std::vector<int64_t>& dims) {
    C_Status status;
    PD_MetaTensorSetDims(raw_data(), dims.size(), dims.data(), &status);
    PD_CHECK_STATUS(status);
  }

  void set_strides(const std::vector<int64_t>& strides) {
    C_Status status;
    PD_MetaTensorSetStrides(
        raw_data(), strides.size(), strides.data(), &status);
    PD_CHECK_STATUS(status);
  }

  void set_dtype(PD_DataType data_type) {
    C_Status status;
    PD_MetaTensorSetDataType(raw_data(), data_type, &status);
    PD_CHECK_STATUS(status);
  }

  void set_layout(PD_DataLayout data_layout) {
    C_Status status;
    PD_MetaTensorSetDataLayout(raw_data(), data_layout, &status);
    PD_CHECK_STATUS(status);
  }
};

}  // namespace capi
}  // namespace phi

#endif
