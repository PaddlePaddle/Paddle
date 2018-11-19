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

#include "paddle/fluid/framework/tvm/tvm_runtime.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace tvm {
namespace runtime {

#ifdef PADDLE_WITH_CUDA
void SetStream(int device_id, const cudaStream_t &stream) {
  ::TVMSetStream(kDLGPU, device_id, stream);
}

void SetStream(const platform::CUDAPlace &place) {
  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  ::TVMSetStream(kDLGPU, place.device, dev_ctx->stream());
}
#endif

namespace internal {
template <typename T>
static DLDataType GetDLDataTypeCode() {
  DLDataType dtype;
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

static DLDataType GetDLDataTypeFromTypeIndex(const std::type_index &type) {
#define REG_TVM_DL_DATA_TYPE(type) \
  { std::type_index(typeid(type)), GetDLDataTypeCode<type>() }
  static const std::unordered_map<std::type_index, DLDataType>
      type_to_dtype_map(
          {REG_TVM_DL_DATA_TYPE(platform::float16), REG_TVM_DL_DATA_TYPE(float),
           REG_TVM_DL_DATA_TYPE(double), REG_TVM_DL_DATA_TYPE(int),
           REG_TVM_DL_DATA_TYPE(int64_t), REG_TVM_DL_DATA_TYPE(bool),
           REG_TVM_DL_DATA_TYPE(size_t), REG_TVM_DL_DATA_TYPE(int16_t),
           REG_TVM_DL_DATA_TYPE(uint8_t), REG_TVM_DL_DATA_TYPE(int8_t)});

  return type_to_dtype_map.at(type);
#undef REG_TVM_DL_DATA_TYPE
}

struct DLContextVisitor : public boost::static_visitor<DLContext> {
  inline DLContext operator()(const platform::CPUPlace &place) const {
    DLContext ctx;
    ctx.device_type = kDLCPU;
    ctx.device_id = 0;
    return ctx;
  }

  inline DLContext operator()(const platform::CUDAPlace &place) const {
#ifdef PADDLE_WITH_CUDA
    DLContext ctx;
    ctx.device_type = kDLGPU;
    ctx.device_id = place.device;
    return ctx;
#else
    PADDLE_THROW("platform::CUDAPlace is not supported in CPU only version");
#endif
  }

  inline DLContext operator()(const platform::CUDAPinnedPlace &place) const {
#ifdef PADDLE_WITH_CUDA
    DLContext ctx;
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
  using TVMDimType = decltype(t_.ndim);  // int
  t_.ndim = static_cast<TVMDimType>(dims.size());

  // init shape, tensor dims
  t_.shape = shape_;
  for (TVMDimType i = 0; i < t_.ndim; ++i) {
    t_.shape[i] = dims[i];
  }

  // init strides, nullptr means the tensor is compact
  t_.strides = nullptr;

  // init byte_offset
  t_.byte_offset = 0;
}

PackedFunc GetFuncFromLib(const std::string &lib_name,
                          const std::string &func_name) {
  auto module = Module::LoadFromFile(lib_name);
  auto func = module.GetFunction(func_name);
  PADDLE_ENFORCE(func != nullptr, "Cannot find function %s from library %s",
                 func_name, lib_name);
  return func;
}

}  // namespace runtime
}  // namespace tvm
}  // namespace framework
}  // namespace paddle
