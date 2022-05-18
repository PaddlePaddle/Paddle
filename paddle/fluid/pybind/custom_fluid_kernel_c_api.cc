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

#if !defined(_WIN32) && !defined(__APPLE__)

#include "paddle/phi/core/custom_fluid_kernel_c_api.h"

#include "paddle/fluid/framework/operator.h"

using Tensor = paddle::framework::Tensor;

using ExecutionContext = paddle::framework::ExecutionContext;

using DeviceContext = paddle::platform::CustomDeviceContext;

inline PD_DataType ToPDDataType(paddle::experimental::DataType dtype) {
#define return_result(in, ret)             \
  case paddle::experimental::DataType::in: \
    return PD_DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataType %d is not supported.", dtype));
    }
  }
#undef return_result
}

inline paddle::experimental::DataType ToPhiDataType(PD_DataType dtype) {
#define return_result(in, ret) \
  case PD_DataType::in:        \
    return paddle::experimental::DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataType %d is not supported.", dtype));
    }
  }
#undef return_result
}

// // convert Tensor to PD_Tensor
// PD_Tensor* PD_TensorFromTensor(const Tensor* src, PD_Status* status);

// // convert PD_Tensor to Tensor
// void PD_TensorToTensor(const PD_Tensor* src, Tensor** dst, PD_Status*
// status);

void PD_GetAttrBool(PD_ExecutionContext* ctx, const char* attr_name,
                    bool* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<bool>(attr_name);
}

void PD_GetAttrInt8(PD_ExecutionContext* ctx, const char* attr_name,
                    int8_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<int8_t>(attr_name);
}

void PD_GetAttrInt16(PD_ExecutionContext* ctx, const char* attr_name,
                     int16_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<int16_t>(attr_name);
}

void PD_GetAttrInt32(PD_ExecutionContext* ctx, const char* attr_name,
                     int32_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<int32_t>(attr_name);
}

void PD_GetAttrInt64(PD_ExecutionContext* ctx, const char* attr_name,
                     int64_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<int64_t>(attr_name);
}

void PD_GetAttrUInt8(PD_ExecutionContext* ctx, const char* attr_name,
                     uint8_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<uint8_t>(attr_name);
}

void PD_GetAttrUInt16(PD_ExecutionContext* ctx, const char* attr_name,
                      uint16_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<uint16_t>(attr_name);
}

void PD_GetAttrUInt32(PD_ExecutionContext* ctx, const char* attr_name,
                      uint32_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<uint32_t>(attr_name);
}

void PD_GetAttrUInt64(PD_ExecutionContext* ctx, const char* attr_name,
                      uint64_t* attr_val, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  *attr_val = exe_ctx->Attr<uint64_t>(attr_name);
}

PD_Stream PD_GetStream(PD_ExecutionContext* ctx, PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  return reinterpret_cast<PD_Stream>(
      exe_ctx->template device_context<DeviceContext>().stream());
}

PD_Tensor* PD_GetInput(PD_ExecutionContext* ctx, const char* name,
                       PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  return reinterpret_cast<PD_Tensor*>(
      const_cast<Tensor*>(exe_ctx->Input<Tensor>(name)));
}

PD_Tensor* PD_GetOutput(PD_ExecutionContext* ctx, const char* name,
                        PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  return reinterpret_cast<PD_Tensor*>(exe_ctx->Output<Tensor>(name));
}

PD_DataType PD_ExpectedOutputDataType(PD_ExecutionContext* ctx,
                                      const char* name, PD_Status* status) {
  PD_DataType dtype = PD_DataType::UNDEFINED;
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  auto tensor = exe_ctx->Output<Tensor>(name);
  if (tensor) {
    dtype = ToPDDataType(tensor->dtype());
  } else {
    *status = C_FAILED;
  }
  return dtype;
}

size_t PD_GetOutputNumDims(PD_ExecutionContext* ctx, const char* name,
                           PD_Status* status) {
  size_t ndims = 0;
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  auto tensor = exe_ctx->Output<Tensor>(name);
  if (tensor) {
    ndims = tensor->dims().size();
  } else {
    *status = C_FAILED;
  }
  return ndims;
}

size_t PD_GetOutputDim(PD_ExecutionContext* ctx, const char* name, size_t index,
                       PD_Status* status) {
  *status = C_SUCCESS;
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  auto tensor = exe_ctx->Output<Tensor>(name);
  size_t dim = 0;
  if (tensor) {
    dim = tensor->dims()[index];
  } else {
    *status = C_FAILED;
  }
  return dim;
}

void PD_AllocateTensor(PD_ExecutionContext* ctx, PD_Tensor* tensor,
                       bool on_host, PD_Status* status) {
  auto cc_tensor = reinterpret_cast<Tensor*>(tensor);
  auto exe_ctx = reinterpret_cast<ExecutionContext*>(ctx);
  auto place = exe_ctx->GetPlace();
  auto dtype = cc_tensor->dtype();
  if (on_host) {
    place = paddle::CPUPlace();
  }
  if (dtype == paddle::experimental::DataType::FLOAT64) {
    cc_tensor->mutable_data<double>(place);
  } else if (dtype == paddle::experimental::DataType::FLOAT32) {
    cc_tensor->mutable_data<float>(place);
  } else if (dtype == paddle::experimental::DataType::FLOAT16) {
    cc_tensor->mutable_data<paddle::float16>(place);
  } else if (dtype == paddle::experimental::DataType::INT8) {
    cc_tensor->mutable_data<int8_t>(place);
  } else if (dtype == paddle::experimental::DataType::INT16) {
    cc_tensor->mutable_data<int16_t>(place);
  } else if (dtype == paddle::experimental::DataType::INT32) {
    cc_tensor->mutable_data<int32_t>(place);
  } else if (dtype == paddle::experimental::DataType::INT64) {
    cc_tensor->mutable_data<int64_t>(place);
  } else if (dtype == paddle::experimental::DataType::UINT8) {
    cc_tensor->mutable_data<uint8_t>(place);
  } else if (dtype == paddle::experimental::DataType::UINT16) {
    // cc_tensor->mutable_data<uint16_t>(place);
    cc_tensor->mutable_data<int16_t>(place);
  } else if (dtype == paddle::experimental::DataType::UINT32) {
    // cc_tensor->mutable_data<uint32_t>(place);
    cc_tensor->mutable_data<int32_t>(place);
  } else if (dtype == paddle::experimental::DataType::UINT64) {
    //   cc_tensor->mutable_data<uint64_t>(place);
    cc_tensor->mutable_data<int64_t>(place);
  } else {
  }
}

PD_DataType PD_TensorType(const PD_Tensor* tensor, PD_Status* status) {
  PD_DataType dtype = PD_DataType::UNDEFINED;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    dtype = ToPDDataType(cc_tensor->dtype());
  }
  return dtype;
}

size_t PD_NumDims(const PD_Tensor* tensor, PD_Status* status) {
  size_t ndims = 0;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    ndims = cc_tensor->dims().size();
  }
  return ndims;
}

size_t PD_Dim(const PD_Tensor* tensor, size_t index, PD_Status* status) {
  size_t dim = 0;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    dim = cc_tensor->dims()[index];
  }
  return dim;
}

size_t PD_TensorByteSize(const PD_Tensor* tensor, PD_Status* status) {
  size_t byte_size = 0;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    byte_size = cc_tensor->memory_size();
  }
  return byte_size;
}

void* PD_TensorData(const PD_Tensor* tensor, PD_Status* status) {
  void* data = nullptr;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    data = const_cast<void*>(cc_tensor->data());
  }
  return data;
}

size_t PD_TensorElementCount(const PD_Tensor* tensor, PD_Status* status) {
  size_t count = 0;
  if (!tensor) {
    *status = C_FAILED;
  } else {
    *status = C_SUCCESS;
    auto cc_tensor = reinterpret_cast<const Tensor*>(tensor);
    count = cc_tensor->numel();
  }
  return count;
}

void PD_SetTensorDims(PD_Tensor* tensor, size_t ndims, const size_t* dims,
                      PD_Status* status) {
  *status = C_SUCCESS;
  auto cc_tensor = reinterpret_cast<Tensor*>(tensor);
  std::vector<int> shape(dims, dims + ndims);
  cc_tensor->Resize(phi::make_ddim(shape));
}

void PD_SetTensorType(PD_Tensor* tensor, PD_DataType dtype, PD_Status* status) {
  *status = C_SUCCESS;
  auto cc_tensor = reinterpret_cast<Tensor*>(tensor);
  auto phi_dtype = ToPhiDataType(dtype);
  cc_tensor->set_type(phi_dtype);
}

void PD_RegisterKernel(const char* kernel, const char* backend,
                       PD_DataType dtype, void (*fn)(PD_ExecutionContext*)) {
  paddle::framework::OpKernelType key(
      paddle::framework::TransToProtoVarType(ToPhiDataType(dtype)),
      paddle::platform::CustomPlace(backend),
      paddle::framework::DataLayout::kAnyLayout,
      paddle::framework::LibraryType::kPlain, 0);
  paddle::framework::OperatorWithKernel::AllOpKernels()[kernel][key] =
      [fn](const paddle::framework::ExecutionContext& ctx) {
        fn(reinterpret_cast<PD_ExecutionContext*>(
            const_cast<paddle::framework::ExecutionContext*>(&ctx)));
      };
}

#endif
